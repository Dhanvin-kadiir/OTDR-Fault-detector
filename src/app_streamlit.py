"""Streamlit dashboard for interactive OTDR fault detection."""

import sys
from pathlib import Path

# Ensure sibling modules (features.py) are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from features import extract_event_candidates, build_features_around_indices, EVENT_NAMES

# ── Security constants ────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
MAX_ROWS = 100_000
REQUIRED_COLUMNS = {"distance_km", "power_db"}
ALLOWED_MODEL_DIRS = {Path("data").resolve()}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="OTDR Fault Detector (v4)", layout="wide")
st.title("OTDR Fault Detector")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("Controls")
model_path_input = st.sidebar.text_input("Model path", "data/rf_model.pkl")
threshold_break_alert = st.sidebar.slider(
    "Break probability alert threshold", 0.3, 0.99, 0.6, 0.01
)

# ── Secure model path validation ─────────────────────────────────────────────
def _validate_model_path(raw: str) -> Path | None:
    """
    Validate the model path against a whitelist of allowed directories.
    Prevents path traversal attacks (e.g. ../../etc/passwd).
    """
    try:
        p = Path(raw).resolve()
    except Exception:
        return None
    if not p.suffix == ".pkl":
        return None
    # Must reside inside an allowed directory
    for allowed in ALLOWED_MODEL_DIRS:
        try:
            p.relative_to(allowed)
            return p
        except ValueError:
            continue
    return None


safe_model_path = _validate_model_path(model_path_input)
if model_path_input and safe_model_path is None:
    st.sidebar.error(
        "Invalid model path. Path must point to a .pkl file inside the data/ directory."
    )

# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(p: str):
    validated = _validate_model_path(p)
    if validated is None:
        return None
    if not validated.exists() or validated.stat().st_size == 0:
        return None
    try:
        return joblib.load(validated)
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None


clf = load_model(model_path_input)

# ── CSV validation ────────────────────────────────────────────────────────────
def _validate_csv(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Validate uploaded CSV for required columns, size, and numeric types.
    Returns (dataframe, error_message). On success, error_message is ''.
    """
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return None, f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not parse CSV: {e}"

    if len(df) > MAX_ROWS:
        return None, f"Too many rows ({len(df):,}). Maximum allowed: {MAX_ROWS:,}."

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return None, f"Missing required column(s): {', '.join(sorted(missing))}."

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                return None, f"Column '{col}' must contain numeric values only."

    if df["distance_km"].isnull().any() or df["power_db"].isnull().any():
        return None, "Columns 'distance_km' and 'power_db' must not contain NaN values."

    if len(df) < 20:
        return None, "Trace too short (< 20 points). Please upload a full OTDR trace."

    return df, ""


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict_events(df: pd.DataFrame) -> pd.DataFrame | None:
    if clf is None:
        return None
    candidates = extract_event_candidates(df)
    feats = build_features_around_indices(df, candidates)
    if feats.empty:
        return pd.DataFrame(columns=["distance_km", "label_name", "prob_break"])

    X = feats.drop(columns=["idx"])
    pred = clf.predict(X)
    out = feats[["distance_km"]].copy()
    out["label_name"] = [EVENT_NAMES.get(int(p), str(p)) for p in pred]

    if hasattr(clf, "predict_proba"):
        cls_map = {c: i for i, c in enumerate(clf.classes_)}
        if 4 in cls_map:
            out["prob_break"] = clf.predict_proba(X)[:, cls_map[4]]
        else:
            none_col = clf.predict_proba(X)[:, cls_map.get(0, 0)]
            step = ((X["delta_lr"].values - 0.2) / 0.8).clip(0, 1)
            out["prob_break"] = (1 - none_col) * step
    else:
        out["prob_break"] = 0.0
    return out


# ── Main UI ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload OTDR CSV (required columns: distance_km, power_db)", type=["csv"]
)

if uploaded is not None:
    df, err = _validate_csv(uploaded)
    if err:
        st.error(f"Upload rejected: {err}")
    else:
        df = df.sort_values("distance_km").reset_index(drop=True)

        # Trace plot
        fig = go.Figure(
            go.Scatter(
                x=df["distance_km"],
                y=df["power_db"],
                mode="lines",
                name="OTDR Power (dB)",
            )
        )
        fig.update_layout(
            xaxis_title="Distance (km)", yaxis_title="Power (dB)", height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        pred_df = predict_events(df)
        if pred_df is None:
            st.info("Model not loaded – check the model path in the sidebar.")
        elif pred_df.empty:
            st.info("No candidate events detected in this trace.")
        else:
            # Only show non-none events plus the top 20 none events
            fault_rows = pred_df[pred_df["label_name"] != "none"]
            none_rows = pred_df[pred_df["label_name"] == "none"].head(20)
            display_df = pd.concat([fault_rows, none_rows]).sort_values("distance_km")

            st.subheader("Event Predictions")
            if len(fault_rows) > 0:
                st.caption(
                    f"Showing {len(fault_rows)} detected fault(s) and 20 sample 'none' points "
                    f"(out of {len(pred_df)} total candidates)."
                )
            st.dataframe(
                display_df[["distance_km", "label_name", "prob_break"]].rename(
                    columns={
                        "distance_km": "Distance (km)",
                        "label_name": "Predicted Event",
                        "prob_break": "P(BREAK)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            if pred_df["prob_break"].max() >= threshold_break_alert:
                st.error(
                    f"Break-like event detected with probability "
                    f"{pred_df['prob_break'].max():.2f}"
                )
            else:
                st.success("No high-probability break detected.")
else:
    st.info("Upload a CSV to analyse (demo files are available in the `data/` folder).")
    st.caption(
        "Expected CSV columns: `distance_km` (float, km), `power_db` (float, dB). "
        f"Max file size: {MAX_FILE_SIZE_MB} MB. Max rows: {MAX_ROWS:,}."
    )
