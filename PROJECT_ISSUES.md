# Project Issue Register

This document records limitations identified during a review of the repository. These items describe the current implementation and recommended follow-up; they are not claims that fixes have already been implemented.

## High priority

### 1. Evaluation split can leak information across traces

**Where:** `src/train.py` and `src/features.py`

The feature builder creates many candidate rows from each trace, but the trainer randomly splits those rows. Candidates from one trace can therefore appear in both training and test sets, making the test metrics an unreliable estimate of performance on entirely new traces.

**Recommended fix:** split original traces by `trace_id` before building candidate features, or use a group-aware split with trace IDs. Keep all candidates from each trace in one partition. Report class counts, precision, recall, F1, and a confusion matrix on held-out traces. Use independently labeled real traces for external validation.

### 2. Serialized model loading trusts pickle content

**Where:** `src/app_streamlit.py`, `_validate_model_path()` and `load_model()`

The path check limits which directory can be selected, but `joblib.load()` deserializes pickle-based content. A malicious or tampered `.pkl` file inside the allowed directory may execute code when loaded. The path check alone does not make model loading safe.

**Recommended fix:** only load artifacts from a trusted source, document this boundary for operators, and consider a non-executable model format where feasible. Do not let untrusted users place model files in the allowed directory.

## Medium priority

### 3. Real-data conversion uses unverified units and geometry

**Where:** `scripts/fetch_real_data.py`

The adapter spaces 30 samples over an assumed 20 km and scales normalized power by SNR. This does not establish that the output represents calibrated distance or power in dB. It also interprets `Location` as a sample index; that assumption needs confirmation against the source schema.

**Recommended fix:** verify source field definitions and units, map location to a sample index only when confirmed by the schema, preserve provenance and conversion metadata, and check the conversion against known traces. Until then, treat the output as exploratory rather than evidence of real-world model accuracy.

### 4. Published performance claims are not reproducible as generalization results

**Where:** previous README accuracy badge and Model Performance table

The previous README reported 98% overall accuracy and 99% break precision/recall without a trace-level holdout protocol. The repository does not include the generated 600-trace training dataset or saved model needed to reproduce the figures. The row-wise split further limits what they demonstrate.

**Recommended fix:** retain performance claims only with a reproducible command, dataset source or generator settings, fixed seed, trace-level split, and full per-class report. Distinguish synthetic validation from independent real-world evaluation.

### 5. Synthetic event overlap and label behavior need validation

**Where:** `src/synthetic_otdr_generator.py`

Event locations are sampled independently, so events may land at or near the same point. Since the label array stores one class per point, later events can overwrite earlier labels while their signal effects remain, creating ambiguous examples.

**Recommended fix:** enforce a minimum separation between generated events or intentionally model and label overlapping events. Add deterministic checks for event counts, label integrity, and signal behavior across seeds.

## Lower priority

### 6. Dashboard predictions are not marked on the trace

**Where:** `src/app_streamlit.py`

The dashboard plots the raw trace and lists candidate predictions separately. It does not show event markers on the plot, making it harder to relate a prediction row to its signal position.

**Recommended fix:** add plot markers for predicted events, distinguish classes visually, and connect the break alert to clearly identified candidate points.

### 7. Local checkout has unrelated uncommitted changes

At review time, the local checkout showed `ai_fiber_faults_commands.txt` deleted and `Proof/` untracked. These were existing local changes and should be reviewed separately before any code or presentation commit. They are not included in this documentation update.

## Suggested order

1. Replace candidate-row random splitting with trace-grouped evaluation.
2. Re-run and document metrics on held-out traces.
3. Verify the real-data schema and conversion before claiming real-world validation.
4. Document and enforce trusted model artifact handling.
5. Improve synthetic event separation and dashboard visualization.
