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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="OTDR Fault Detector (v4)", layout="wide")
st.title("🔍 OTDR Fault Detector")

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Controls")
model_path = st.sidebar.text_input("Model path", "data/rf_model.pkl")
threshold_break_alert = st.sidebar.slider(
    "Break probability alert threshold", 0.3, 0.99, 0.6, 0.01
)


# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(p: str):
    pth = Path(p)
    if not pth.exists() or pth.stat().st_size == 0:
        return None
    try:
        return joblib.load(pth)
    except Exception as e:
        st.warning(f"Could not load model at {pth}: {e}")
        return None


clf = load_model(model_path)


# ── Prediction logic ─────────────────────────────────────────────────────────
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
    "Upload OTDR CSV (columns: distance_km, power_db)", type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded).sort_values("distance_km")

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
        st.subheader("Event Predictions")
        st.dataframe(
            pred_df[["distance_km", "label_name", "prob_break"]].rename(
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
                f"⚠️  Break-like event detected with probability "
                f"{pred_df['prob_break'].max():.2f}"
            )
        else:
            st.success("✅  No high-probability break detected.")
else:
    st.info("Upload a CSV to analyse (demo files are available in the `data/` folder).")
