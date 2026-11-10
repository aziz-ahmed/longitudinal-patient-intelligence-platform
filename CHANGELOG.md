# Changelog

## 0.1.0 — 2026-11-10

- Initial public portfolio release.
- Synthetic longitudinal data generator (seeded, no PHI).
- 78 temporal features, temporal holdout, XGBoost / logistic regression / HistGradientBoosting.
- Calibration, subgroup evaluation, SHAP explainability.
- Verified: 30-day XGBoost 0.734 AUROC / 0.518 AUPRC; 90-day XGBoost 0.754 AUROC / 0.758 AUPRC (n=2000, seed 42, synthetic data).
