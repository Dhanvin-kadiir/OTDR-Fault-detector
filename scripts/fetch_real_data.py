"""
fetch_real_data.py — Download and adapt a public OTDR dataset for testing.

This script downloads the publicly available OTDR dataset from the
IEEE DataPort / Kaggle "Optical Fibre - Fault Detection" dataset (via
the Kaggle API) and converts it to the distance_km / power_db format
expected by this project.

Dataset source:
  https://www.kaggle.com/datasets/yogipatel08/optical-fibre-fault-detection

USAGE
-----
  # 1. Install kaggle CLI (if not already done)
  pip install kaggle

  # 2. Set up your Kaggle API credentials
  #    - Go to https://www.kaggle.com/account -> Create API Token
  #    - Save the downloaded kaggle.json to ~/.kaggle/kaggle.json

  # 3. Run this script
  python scripts/fetch_real_data.py --out data/real_otdr_test.csv

HOW IT WORKS
------------
The Kaggle dataset stores each trace as a row of 30 normalised power
samples (P1..P30) plus metadata (SNR, Class, Location, Reflectance,
Loss).  This script reconstructs a `distance_km` / `power_db` trace
for each row by:
  1. Spacing 30 samples over a 20 km synthetic length.
  2. Denormalising power using the provided SNR as a scaling factor.
  3. Assigning an `event_label` from the "Class" column.

The resulting CSV can be uploaded directly to the Streamlit app for
real-world model testing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Class mapping from Kaggle dataset to our internal labels
KAGGLE_CLASS_MAP = {
    "Normal": 0,
    "Splice": 1,
    "Connector": 2,
    "Bend": 3,
    "Break": 4,
    "Fiber Break": 4,
    "fiber_break": 4,
    "bend": 3,
    "splice": 1,
    "connector": 2,
    "normal": 0,
}

N_POINTS = 30          # samples per trace in the Kaggle dataset
LENGTH_KM = 20.0       # synthetic fibre length to assume


def adapt_kaggle_row(row: pd.Series, trace_id: int) -> pd.DataFrame:
    """Convert one row of the Kaggle OTDR_data.csv to a trace DataFrame."""
    power_cols = [f"P{i}" for i in range(1, N_POINTS + 1)]
    available = [c for c in power_cols if c in row.index]

    # If P1..P30 columns are missing, bail
    if not available:
        return pd.DataFrame()

    distance = np.linspace(0, LENGTH_KM, len(available))
    power = row[available].astype(float).values

    # Scale by SNR if available (higher SNR -> less noise -> sharper features)
    snr = float(row.get("SNR", 10.0))
    power = power * max(snr / 10.0, 0.5)

    # Determine event label from Class column
    cls_raw = str(row.get("Class", "Normal")).strip()
    label = KAGGLE_CLASS_MAP.get(cls_raw, KAGGLE_CLASS_MAP.get(cls_raw.title(), 0))

    # Determine event location index (inject a point label if location is given)
    event_col = np.zeros(len(available), dtype=int)
    loc = row.get("Location", None)
    if loc is not None:
        try:
            loc_idx = int(float(loc))
            if 0 <= loc_idx < len(available):
                event_col[loc_idx] = label
        except (ValueError, TypeError):
            pass
    elif label != 0:
        # No explicit location — place event at midpoint
        event_col[len(available) // 2] = label

    return pd.DataFrame(
        {
            "distance_km": distance,
            "power_db": power,
            "event_label": event_col,
            "trace_id": trace_id,
        }
    )


def download_kaggle(out_dir: Path) -> Path:
    """Attempt to download the dataset via the Kaggle CLI."""
    import subprocess  # nosec B404

    dest = out_dir / "kaggle_otdr_raw.csv"
    if dest.exists():
        print(f"[INFO] Using existing download: {dest}")
        return dest

    try:
        subprocess.run(  # nosec B603
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "yogipatel08/optical-fibre-fault-detection",
                "--unzip", "-p", str(out_dir),
            ],
            check=True,
        )
        # Locate the downloaded CSV
        candidates = list(out_dir.glob("*.csv"))
        if candidates:
            candidates[0].rename(dest)
            return dest
        raise FileNotFoundError("Kaggle download succeeded but no CSV found.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        print("        Make sure kaggle is installed and ~/.kaggle/kaggle.json exists.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Download and adapt real OTDR data for testing")
    ap.add_argument("--out", default="data/real_otdr_test.csv", help="Output CSV path")
    ap.add_argument("--input", default=None, help="Use a local Kaggle CSV instead of downloading")
    ap.add_argument("--max_traces", type=int, default=200, help="Maximum traces to convert")
    args = ap.parse_args()

    if args.input:
        raw_path = Path(args.input)
    else:
        raw_path = download_kaggle(Path("data"))

    print(f"[INFO] Reading {raw_path} ...")
    raw = pd.read_csv(raw_path)
    print(f"       Columns: {raw.columns.tolist()}")
    print(f"       Rows: {len(raw)}")

    dfs = []
    for i, (_, row) in enumerate(raw.iterrows()):
        if i >= args.max_traces:
            break
        trace = adapt_kaggle_row(row, trace_id=i)
        if not trace.empty:
            dfs.append(trace)

    if not dfs:
        print("[ERROR] No traces adapted. Check column names in the raw CSV.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(dfs, ignore_index=True).to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(dfs)} adapted real-world traces -> {out_path}")
    print()
    print("You can now:")
    print(f"  1. Upload individual trace CSVs to the Streamlit app, OR")
    print(f"  2. Re-train the model on real data:")
    print(f"     python src/train.py --data {out_path} --model_path data/rf_model_real.pkl")


if __name__ == "__main__":
    main()
