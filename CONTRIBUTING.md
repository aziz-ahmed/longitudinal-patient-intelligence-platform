# Contributing

This is a portfolio project. Suggestions and small fixes are welcome via pull request.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m lpip.pipeline --n-patients 300 --seed 42 --output-dir outputs-smoke
```

## Rules

- **Synthetic data only.** Never add real patient data, client code, credentials, or PHI.
- Keep the temporal split intact — do not replace it with a random split.
- Add or update a test when you change data generation, features, or evaluation.
- Keep the README results honest: report the metrics the pipeline actually produces on synthetic data, with seed and sample size.
