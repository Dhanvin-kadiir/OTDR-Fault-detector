"""Generate synthetic OTDR traces with realistic fibre-optic fault signatures."""

import argparse
import numpy as np
import pandas as pd

EVENT_NONE, EVENT_SPLICE, EVENT_CONNECTOR, EVENT_BEND, EVENT_BREAK = 0, 1, 2, 3, 4


def simulate_trace(
    num_points: int = 4000,
    length_km: float = 20.0,
    base_loss_db: float = 0.22,
    noise_std: float = 0.03,
    n_splices: tuple = (1, 3),
    n_connectors: tuple = (1, 3),
    n_bends: tuple = (1, 2),
    break_prob: float = 1.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return a single synthetic OTDR trace as a DataFrame."""
    if rng is None:
        rng = np.random.default_rng()

    distance = np.linspace(0, length_km, num_points)
    slope = -base_loss_db + rng.normal(0, 0.02)
    power = slope * distance + rng.normal(0, noise_std, size=num_points)
    events = np.full(num_points, EVENT_NONE, dtype=int)

    def _place():
        return rng.integers(120, num_points - 120)

    # -- splices --
    for _ in range(rng.integers(n_splices[0], n_splices[1] + 1)):
        idx = _place()
        step = rng.uniform(0.08, 0.25)
        power[idx:] -= step
        events[idx] = EVENT_SPLICE

    # -- connectors --
    for _ in range(rng.integers(n_connectors[0], n_connectors[1] + 1)):
        idx = _place()
        spike = rng.uniform(0.15, 0.45)
        power[idx] += spike
        power[idx + 1 :] -= spike * 0.4
        events[idx] = EVENT_CONNECTOR

    # -- bends --
    for _ in range(rng.integers(n_bends[0], n_bends[1] + 1)):
        idx = _place()
        extra = rng.uniform(0.03, 0.08)
        power[idx:] -= extra
        events[idx] = EVENT_BEND

    # -- break --
    if rng.random() < break_prob:
        idx = rng.integers(int(num_points * 0.45), int(num_points * 0.9))
        floor = np.min(power) - rng.uniform(3.0, 5.0)
        power[idx:] = floor + rng.normal(0, noise_std * 3, num_points - idx)
        events[idx] = EVENT_BREAK

    return pd.DataFrame(
        {"distance_km": distance, "power_db": power, "event_label": events}
    )


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic OTDR traces")
    ap.add_argument("--n_traces", type=int, default=650)
    ap.add_argument("--out", type=str, default="data/otdr_traces.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dfs = []
    for tid in range(args.n_traces):
        df = simulate_trace(rng=rng)
        df["trace_id"] = tid
        dfs.append(df)

    pd.concat(dfs, ignore_index=True).to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.n_traces} synthetic traces -> {args.out}")


if __name__ == "__main__":
    main()
