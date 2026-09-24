# OTDR Fault Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Model evaluation](https://img.shields.io/badge/evaluation-synthetic%20prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**An AI-powered Optical Time-Domain Reflectometer (OTDR) trace analyser that detects fibre-optic faults in real time using a trained Random Forest classifier.**

[Features](#features) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage) · [Real-World Data](#real-world-data) · [Security](#security) · [Model Evaluation](#model-evaluation) · [Known Issues](#known-issues)

</div>

---

## Overview

OTDR traces reflect the health of optical fibre cables by measuring back-scattered light over the cable's length. Anomalies in the trace — such as sudden power drops or reflective spikes — indicate faults.

This project automates fault detection by:

1. **Generating** realistic synthetic OTDR traces with ground-truth labels.
2. **Extracting** hand-crafted signal features around candidate event locations.
3. **Training** a Random Forest classifier to distinguish between fault types.
4. **Serving** predictions through a secure, interactive Streamlit web dashboard.

---

## Features

| Capability | Detail |
| --- | --- |
| Fault types detected | None, Splice, Connector, Bend, **Break** |
| ML model | Random Forest (450 estimators, class-balanced) |
| Feature engineering | Gradient-based candidate detection + local window features |
| Dashboard | Interactive Plotly trace viewer + event prediction table |
| Alert system | Configurable break-probability threshold with visual alerts |
| Data generation | Configurable synthetic OTDR simulator (`break_prob`, `n_traces`, `seed`) |
| Security | Input validation, path traversal prevention, file size & row limits |
| Real-world data | Adapter for [Kaggle OTDR dataset](https://www.kaggle.com/datasets/yogipatel08/optical-fibre-fault-detection) |

---

## Architecture

```text
synthetic_otdr_generator.py
       |
       v
data/otdr_traces.csv  (distance_km, power_db, event_label, trace_id)
       |
       v
features.py  -- extract_event_candidates()
             -- build_features_around_indices()
             -- label_for_index()  (exact-point GT labeling)
       |
       v
train.py  -->  data/rf_model.pkl
       |
       v
app_streamlit.py  (Upload CSV -> Visualise -> Predict -> Alert)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Dhanvin-kadiir/OTDR-Fault-detector.git
cd OTDR-Fault-detector

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Synthetic Training Data

```bash
python src/synthetic_otdr_generator.py --n_traces 600 --out data/otdr_traces.csv --break_prob 0.5
```

| Argument | Default | Description |
| --- | --- | --- |
| `--n_traces` | `600` | Number of traces to generate |
| `--out` | `data/otdr_traces.csv` | Output CSV path |
| `--break_prob` | `0.5` | Probability each trace has a break event |
| `--seed` | `42` | Random seed for reproducibility |

### 2. Train the Model

```bash
python src/train.py --data data/otdr_traces.csv --model_path data/rf_model.pkl
```

| Argument | Default | Description |
| --- | --- | --- |
| `--data` | *(required)* | Path to the training CSV |
| `--model_path` | `data/rf_model.pkl` | Output model path |

### 3. Launch the Dashboard

```bash
streamlit run src/app_streamlit.py
```

Then open **<http://localhost:8501>** in your browser.

Upload any OTDR CSV with columns `distance_km` and `power_db` (demo files in `data/`).

---

## Real-World Data

The project includes a utility to download and adapt the publicly available [Kaggle OTDR dataset](https://www.kaggle.com/datasets/yogipatel08/optical-fibre-fault-detection) (IEEE DataPort, 74 MB) for real-world model testing.

### Setup

```bash
# Install the Kaggle CLI
pip install kaggle

# Download your API token from https://www.kaggle.com/account
# Place it at: ~/.kaggle/kaggle.json   (Linux/macOS)
#              C:\Users\<you>\.kaggle\kaggle.json   (Windows)
```

### Download and Convert

```bash
python scripts/fetch_real_data.py --out data/real_otdr_test.csv --max_traces 200
```

| Argument | Default | Description |
| --- | --- | --- |
| `--out` | `data/real_otdr_test.csv` | Output CSV path |
| `--input` | *(auto-download)* | Path to local Kaggle CSV (skips download) |
| `--max_traces` | `200` | Max number of traces to convert |

The adapter maps Kaggle P1..P30 normalized power values into the app's expected columns. It currently assumes a 20 km distance span and scales values using SNR; this is an exploratory format conversion, not a validated physical conversion to calibrated OTDR distance and dB. Verify the source schema, units, and location encoding before using its output for model evaluation or operational decisions.

---

## Security

The dashboard applies basic upload checks. A previous Bandit scan is not a guarantee that the application is secure; rerun it against the current code when making a release.

### Security measures in `app_streamlit.py`

| Threat | Mitigation |
| --- | --- |
| Path traversal when selecting a model | Model path is restricted to the `data/` directory |
| Malicious large file upload (DoS) | File size cap: 10 MB |
| Memory exhaustion via oversized CSV | Row count cap: 100,000 rows |
| Invalid/malformed input data | Column validation, numeric type checking, NaN detection |
| Excessively short traces | Minimum 20 data points enforced |

To re-run the security scan:

```bash
pip install bandit
bandit -r src/ scripts/
```

**Trust boundary:** model files are loaded with `joblib`, which uses Python pickle serialization. Loading an untrusted or tampered model file can execute code. Only load model files from a trusted source; restricting the path does not make a malicious pickle safe.

## Model Evaluation

The training script prints a random candidate-row holdout report. Candidates from the same trace can appear in both training and test sets, so these metrics do not establish performance on unseen traces. The previously quoted 98% accuracy and 99% break precision/recall have not been validated with a trace-level holdout and should not be treated as verified performance.

For a meaningful estimate, split by `trace_id` so every test trace is excluded from training, document the dataset and seed, and report per-class metrics plus the confusion matrix. Validate on independently collected and labeled real OTDR traces before making operational claims. See [PROJECT_ISSUES.md](PROJECT_ISSUES.md) for the remediation list.

---

## Dataset

Synthetic traces are generated via `src/synthetic_otdr_generator.py`. Each trace simulates:

- **Background attenuation** — linear power loss along the fibre.
- **Splices** — sudden step-down losses.
- **Connectors** — localised reflective spikes.
- **Bends** — gradual additional attenuation.
- **Breaks** — catastrophic power drop with a noise floor.

Output CSV schema:

| Column | Description |
| --- | --- |
| `distance_km` | Distance along the fibre (km) |
| `power_db` | Back-scattered power (dB) |
| `event_label` | Ground-truth label (0–4) |
| `trace_id` | Unique identifier per trace |

---

## Project Structure

```text
OTDR-Fault-detector/
├── data/
│   ├── faulty_trace.csv          # Example trace with faults
│   └── healthy_trace.csv         # Example fault-free trace
├── scripts/
│   └── fetch_real_data.py        # Download & adapt Kaggle OTDR dataset
├── src/
│   ├── app_streamlit.py          # Secure Streamlit web dashboard
│   ├── features.py               # Feature extraction & dataset builder
│   ├── synthetic_otdr_generator.py  # Synthetic trace generator
│   └── train.py                  # Model training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Requirements

```text
numpy
pandas
scikit-learn
scipy
streamlit
plotly
joblib
```

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Dhanvin Kadiir** — [GitHub](https://github.com/Dhanvin-kadiir)

## Known Issues

See [PROJECT_ISSUES.md](PROJECT_ISSUES.md) for known limitations, impact, and recommended fixes. The main open items are trace-level model evaluation, validation of the real-data conversion, and trusted handling of serialized model files.
