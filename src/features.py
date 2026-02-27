"""Feature engineering utilities for OTDR trace event detection."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

EVENT_NAMES = {0: "none", 1: "splice", 2: "connector", 3: "bend", 4: "break"}


def extract_event_candidates(
    df_trace: pd.DataFrame,
    prominence: float = 0.02,
    width: int = 1,
    neg_k: float = 0.8,
) -> np.ndarray:
    """Return indices of candidate event locations in a single trace."""
    p = df_trace["power_db"].values
    x = df_trace["distance_km"].values

    d = np.gradient(p, x)
    neg = np.where(d < (np.mean(d) - neg_k * np.std(d)))[0]
    peaks, _ = find_peaks(p, prominence=prominence, width=width)

    candidates = np.unique(np.concatenate([neg, peaks]))
    return candidates[(candidates > 5) & (candidates < len(p) - 5)]


def build_features_around_indices(
    df_trace: pd.DataFrame,
    idxs: np.ndarray,
    window: int = 10,
) -> pd.DataFrame:
    """Build engineered features for each candidate index."""
    rows = []
    p = df_trace["power_db"].values
    x = df_trace["distance_km"].values

    for idx in idxs:
        i0 = max(0, idx - window)
        i1 = min(len(p), idx + window + 1)
        seg = p[i0:i1]
        segx = x[i0:i1]
        slope = np.polyfit(segx - segx[0], seg, 1)[0] if len(seg) > 2 else 0.0
        lm = np.mean(p[max(0, idx - 5) : idx]) if idx > 0 else p[0]
        rm = np.mean(p[idx + 1 : min(len(p), idx + 6)]) if idx < len(p) - 1 else p[-1]
        rows.append(
            {
                "idx": idx,
                "distance_km": x[idx],
                "mean_power": float(np.mean(seg)),
                "std_power": float(np.std(seg)),
                "range_power": float(np.ptp(seg)),
                "delta_lr": float(lm - rm),
                "slope_local": float(slope),
            }
        )
    return pd.DataFrame(rows)


def label_from_window(df_trace: pd.DataFrame, idx: int, window: int = 3) -> int:
    """Majority-vote label around *idx* in the ground-truth column."""
    i0 = max(0, idx - window)
    i1 = min(len(df_trace), idx + window + 1)
    lab = df_trace["event_label"].values[i0:i1]
    vals, cnts = np.unique(lab, return_counts=True)
    return int(vals[np.argmax(cnts)])


def make_dataset(
    df_all: pd.DataFrame,
    max_candidates_per_trace: int = 160,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) from a multi-trace CSV with columns trace_id, distance_km, power_db, event_label."""
    Xs, ys = [], []
    for _, t in df_all.groupby("trace_id"):
        c = extract_event_candidates(t, prominence=0.02, width=1, neg_k=0.8)
        gt_idx = np.where(t["event_label"].values != 0)[0]
        c = np.unique(np.concatenate([c, gt_idx]))
        if len(c) == 0:
            continue
        if len(c) > max_candidates_per_trace:
            c = np.random.choice(c, size=max_candidates_per_trace, replace=False)
        f = build_features_around_indices(t, c)
        labels = [label_from_window(t, int(i)) for i in f["idx"].values]
        Xs.append(f.drop(columns=["idx"]))
        ys.append(pd.Series(labels, name="label"))
    if not Xs:
        return pd.DataFrame(), pd.Series(dtype=int)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)
