# Models — notes

## Tabular models (default)

- **Logistic Regression** — scaled, class-weighted. Interpretable baseline.
- **XGBoost** — primary model, 300 trees, depth 4.
- **HistGradientBoosting** — sklearn fallback if XGBoost is unavailable.

## Sequence models (extension)

The original project also benchmarked LSTM and Transformer models on
raw event sequences. That is the natural next step here:

1. Use `lpip.models.build_sequences()` to build per-patient lab sequences.
2. Pad sequences, add a mask, and train a small PyTorch LSTM with a
   binary head for 30-day hospitalization.
3. Compare AUROC/AUPRC against the tabular XGBoost model on the *same*
   temporal split — do not re-split randomly.

PyTorch is intentionally not a hard dependency, so this repo runs on
CPU-only machines without a large install.

## Why a temporal split?

A random split would let a patient's future leak into training via
nearby index dates. Splitting by index date mimics prospective
deployment: train on the past, test on the future.
