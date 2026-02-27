# OTDR Fault Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**An AI-powered Optical Time-Domain Reflectometer (OTDR) trace analyser that detects fibre-optic faults in real time using a trained Random Forest classifier.**

[Features](#features) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage) · [Model Performance](#model-performance) · [Project Structure](#project-structure)

</div>

---

## Overview

OTDR traces reflect the health of optical fibre cables by measuring back-scattered light over the cable's length. Anomalies in the trace — such as sudden power drops or reflective spikes — indicate faults.

This project automates fault detection by:

1. **Generating** realistic synthetic OTDR traces with ground-truth labels.
2. **Extracting** hand-crafted features around candidate event locations.
3. **Training** a Random Forest classifier to distinguish between fault types.
4. **Serving** predictions through an interactive Streamlit web dashboard.

---

## Features

| Capability | Detail |
|---|---|
| Fault types detected | None, Splice, Connector, Bend, **Break** |
| ML model | Random Forest (450 estimators, class-balanced) |
| Feature engineering | Gradient-based candidate detection + local window features |
| Dashboard | Interactive Plotly trace viewer + event prediction table |
| Alert system | Configurable break-probability threshold with visual alerts |
| Data generation | Configurable synthetic OTDR simulator (break_prob, n_traces, seed) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Pipeline Overview                      │
│                                                          │
│  synthetic_otdr_generator.py                             │
│       └─> data/otdr_traces.csv                           │
│              |                                           │
│  features.py  (extract_event_candidates + build_features)│
│              |                                           │
│  train.py  --> data/rf_model.pkl                         │
│              |                                           │
│  app_streamlit.py  (Upload CSV -> Visualise -> Predict)  │
└──────────────────────────────────────────────────────────┘
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
|---|---|---|
| `--n_traces` | `600` | Number of traces to generate |
| `--out` | `data/otdr_traces.csv` | Output CSV path |
| `--break_prob` | `0.5` | Probability each trace has a break event |
| `--seed` | `42` | Random seed for reproducibility |

### 2. Train the Model

```bash
python src/train.py --data data/otdr_traces.csv --model_path data/rf_model.pkl
```

| Argument | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to the training CSV |
| `--model_path` | `data/rf_model.pkl` | Output model path |

### 3. Launch the Dashboard

```bash
streamlit run src/app_streamlit.py
```

Then open **<http://localhost:8501>** in your browser.

Upload any OTDR CSV with columns `distance_km` and `power_db` (demo files are in `data/`).

---

## Model Performance

The model is trained on 600 balanced synthetic traces (50% with breaks) and achieves the following on a held-out test set:

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 — None | 0.99 | 1.00 | 0.99 |
| 1 — Splice | 0.91 | 0.79 | 0.85 |
| 2 — Connector | 0.97 | 0.77 | 0.86 |
| 3 — Bend | 0.81 | 0.51 | 0.62 |
| **4 — Break** | **0.99** | **0.99** | **0.99** |
| **Overall accuracy** | | | **98%** |

Break events (the most critical fault) achieve 99% precision and 99% recall.

---

## Dataset

Synthetic traces are generated via `src/synthetic_otdr_generator.py`. Each trace simulates:

- **Background attenuation** — linear power loss along the fibre.
- **Splices** — sudden step-down losses.
- **Connectors** — localised reflective spikes.
- **Bends** — gradual additional attenuation.
- **Breaks** — catastrophic power drop with a noise floor.

The output CSV schema is:

| Column | Description |
|---|---|
| `distance_km` | Distance along the fibre (km) |
| `power_db` | Back-scattered power (dB) |
| `event_label` | Ground-truth label (0–4) |
| `trace_id` | Unique identifier per trace |

---

## Project Structure

```
OTDR-Fault-detector/
├── data/
│   ├── faulty_trace.csv          # Example trace with faults
│   ├── healthy_trace.csv         # Example fault-free trace
│   ├── otdr_traces.csv           # Training dataset (generated)
│   └── rf_model.pkl              # Trained classifier (generated)
├── src/
│   ├── app_streamlit.py          # Streamlit web dashboard
│   ├── features.py               # Feature extraction & dataset builder
│   ├── synthetic_otdr_generator.py  # Synthetic trace generator
│   └── train.py                  # Model training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Requirements

```
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

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Dhanvin Kadiir** — [GitHub](https://github.com/Dhanvin-kadiir)
