# Longitudinal Patient Intelligence Platform — Synthetic Portfolio Version

[![CI](https://github.com/aziz-ahmed/longitudinal-patient-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aziz-ahmed/longitudinal-patient-intelligence-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A clean-room, portfolio rebuild of the ideas behind a longitudinal patient-intelligence project from my resume. **No client code, no client data, and no real patient data are used here.** Everything is synthetic, generated from scratch, and clearly labelled as a demonstration.

Predict 30-day and 90-day hospitalization / clinical deterioration from a patient's longitudinal history — encounters, labs, medications, and vitals — using temporal features, a temporal holdout, calibration, subgroup evaluation, and SHAP explainability.

This mirrors the resume project in *approach*, not in data or code:

- Longitudinal patient representations from EHR-like events
- 50+ temporal features: trajectories, treatment response, utilization, recency, volatility, and rate of deterioration
- Benchmarked XGBoost vs. logistic-regression and gradient-boosting baselines
- Temporal holdout validation, calibration analysis, and subgroup evaluation
- SHAP-based explainability, translated into interpretable risk drivers

**Do not use this for clinical decisions.** It is a portfolio / methods demonstration on synthetic data. The metrics reported here are on synthetic data and are *not* the metrics from the original client project.

## What it does

```
synthetic EHR-like events
  → temporal feature engineering (50+ features)
  → temporal train / test split (by index date, no leakage)
  → XGBoost / Logistic Regression / HistGradientBoosting
  → AUROC / AUPRC, recall, calibration, subgroup checks
  → SHAP / permutation importance
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data, build features, train, and evaluate
python -m lpip.pipeline --n-patients 2000 --seed 42 --output-dir outputs

# See the results
cat outputs/metrics.json
ls outputs/figures/
```

A fast smoke run:

```bash
python -m lpip.pipeline --n-patients 300 --seed 42 --output-dir outputs-smoke
pytest -q
```

## Project layout

```
src/lpip/
  data.py       # Synthetic longitudinal data generator (seeded, no PHI)
  features.py   # 50+ temporal features from encounters / labs / meds / vitals
  models.py     # Logistic Regression, XGBoost, HistGradientBoosting; LSTM hook (optional)
  evaluate.py   # AUROC/AUPRC, calibration, subgroup evaluation
  explain.py    # SHAP (when installed) with permutation-importance fallback
  pipeline.py   # End-to-end CLI: generate → features → temporal split → train → evaluate
scripts/
  generate_data.py   # Save synthetic events to CSV
  train.py           # Train from saved CSVs
tests/
  test_data.py test_features.py test_pipeline.py
notebooks/
  01_walkthrough.ipynb   # Guided walkthrough on a small synthetic sample
```

## Data — synthetic by design

`src/lpip/data.py` generates:

- `patients`: age, sex, comorbidity count, index date, 30-day and 90-day hospitalization labels
- `encounters`: outpatient / ED / inpatient events in the 180 days before index
- `labs`: repeated measurements (creatinine, haemoglobin, BNP, CRP, glucose) with patient-specific trends
- `medications`: medication count and changes
- `vitals`: repeated systolic BP / heart-rate measurements

Labels are generated from a latent risk score (age, comorbidity, deteriorating labs, recent utilization) plus noise, so the signal is learnable but not deterministic. No real names, dates of birth, MRNs, or clinical notes are present — patient IDs are `SYN-00001`, etc.

## Features

See `src/lpip/features.py` for the full list. Groups include:

- Demographics & comorbidity
- Utilization in 30 / 90 / 180-day windows, recency of last encounter / ED visit
- Lab trajectories per analyte: last value, mean, std / volatility, min / max, slope, 90-day change, abnormal-result count
- Medication burden and changes
- Vital-sign trajectories: last, mean, volatility, slope

## Validation — temporal, not random

Patients are split by `index_date`, not at random:

- Train: index date before the cutoff (default: median index date)
- Test: index date on / after the cutoff

This mimics prospective deployment and avoids leaking future information into training. The pipeline also reports calibration (Brier score, calibration curve data) and subgroup AUROC / recall by age group, sex, and comorbidity burden.

## Models

| Model | Role |
|---|---|
| Logistic Regression (scaled) | Interpretable baseline |
| XGBoost | Primary model |
| HistGradientBoosting | sklearn fallback if XGBoost is unavailable |

An LSTM / Transformer sequence model is the natural next step for the raw event sequences — the sequence builder is in `src/lpip/models.py` (`build_sequences`) and is intentionally left as an optional extension, so this repo runs without PyTorch. See `docs/MODELS.md` for notes.

## Results (synthetic data, seed 42, n=2000)

Verified by running the pipeline in this repo. Results vary with the seed and sample size — run it to reproduce the current numbers in `outputs/metrics.json`.

| Horizon | Model | AUROC | AUPRC |
|---|---|---|---|
| 30-day | Logistic Regression | 0.741 | 0.542 |
| 30-day | XGBoost | 0.734 | 0.518 |
| 30-day | HistGradientBoosting | 0.692 | 0.460 |
| 90-day | XGBoost | 0.754 | 0.758 |
| 90-day | Logistic Regression | 0.751 | 0.767 |
| 90-day | HistGradientBoosting | 0.737 | 0.745 |

Temporal split at 2024-03-23, train n=1000, test n=1000, 78 features. 30-day prevalence in test was 26.9%.

Do not compare these numbers to the original client project — different data, different task definition, different prevalence.

## Limitations

- Synthetic data is smoother and cleaner than real EHR data: no missing-not-at-random patterns, coding errors, or site effects are faithfully reproduced.
- No clinical notes, imaging, or free text are modelled here.
- No fairness, safety, or regulatory claim is made. Subgroup checks are descriptive, not a bias audit.
- Thresholds in the pipeline are illustrative; a deployed system would need clinical review, prospective validation, and governance.

## Author

Mohammed Aziz Ahmed — Data Scientist
- LinkedIn: https://www.linkedin.com/in/mohammed-aziz-ahmed/
- GitHub: https://github.com/aziz-ahmed

## Licence

MIT — see `LICENSE`.
