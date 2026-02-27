"""Feature engineering utilities for OTDR trace event detection."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

EVENT_NAMES = {0: "none", 1: "splice", 2: "connector", 3: "bend", 4: "break"}


def extract_event_candidates(
    df_trace: pd.DataFrame,
    prominence: float = 0.02,
    width: int = 1,
    neg_k: float = 0.6,
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


def label_for_index(df_trace: pd.DataFrame, idx: int) -> int:
    """
    Return the ground-truth label for a candidate index.

    Priority:
      1. Exact point label if non-zero.
      2. Nearest non-zero label within ±5 points.
      3. 0 (none) as fallback.
    """
    labels = df_trace["event_label"].values
    # 1. Exact match
    if labels[idx] != 0:
        return int(labels[idx])
    # 2. Nearest non-zero within window
    lo, hi = max(0, idx - 5), min(len(labels), idx + 6)
    window_labels = labels[lo:hi]
    non_zero = window_labels[window_labels != 0]
    if len(non_zero) > 0:
        vals, cnts = np.unique(non_zero, return_counts=True)
        return int(vals[np.argmax(cnts)])
    return 0


def make_dataset(
    df_all: pd.DataFrame,
    max_candidates_per_trace: int = 160,
) -> tuple:
    """Build (X, y) from a multi-trace CSV with columns trace_id, distance_km, power_db, event_label."""
    Xs, ys = [], []
    for _, t in df_all.groupby("trace_id"):
        t = t.reset_index(drop=True)

        # Candidate positions from signal analysis
        c = extract_event_candidates(t)

        # Always include ALL actual ground-truth event positions
        gt_idx = np.where(t["event_label"].values != 0)[0]
        c = np.unique(np.concatenate([c, gt_idx]))

        if len(c) == 0:
            continue
        if len(c) > max_candidates_per_trace:
            # Keep all GT events + random sample of the rest
            non_gt = np.setdiff1d(c, gt_idx)
            n_extra = max(0, max_candidates_per_trace - len(gt_idx))
            if len(non_gt) > n_extra:
                non_gt = np.random.choice(non_gt, size=n_extra, replace=False)
            c = np.unique(np.concatenate([gt_idx, non_gt]))

        f = build_features_around_indices(t, c)
        labels = [label_for_index(t, int(i)) for i in f["idx"].values]
        Xs.append(f.drop(columns=["idx"]))
        ys.append(pd.Series(labels, name="label"))

    if not Xs:
        return pd.DataFrame(), pd.Series(dtype=int)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)
