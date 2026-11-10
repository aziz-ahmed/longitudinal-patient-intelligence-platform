"""Models: interpretable baseline, XGBoost, and a gradient-boosting fallback.

An LSTM / Transformer on raw event sequences is the natural extension
for this task (benchmarked in the original project). The sequence
builder below is included so that extension is straightforward, but
PyTorch is *not* required to run this repo — the tabular models are
the default and are fully reproducible on CPU.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42) -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "hist_gb": HistGradientBoostingClassifier(random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5,
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:  # pragma: no cover - XGBoost is optional
        pass
    return models


def build_sequences(labs, max_len: int = 20):
    """Build per-patient lab sequences for a sequence model (optional).

    Returns a dict patient_id -> array of shape (T, n_labs), most recent last.
    This is a hook for an LSTM/Transformer extension; it is not used by
    the default tabular pipeline.
    """
    import pandas as pd

    lab_names = sorted(labs["lab_name"].unique())
    seqs = {}
    for pid, group in labs.groupby("patient_id"):
        piv = group.pivot_table(
            index="event_date", columns="lab_name", values="lab_value", aggfunc="mean"
        ).sort_index()
        piv = piv.reindex(columns=lab_names)
        arr = piv.to_numpy(dtype=float)
        # Forward-fill, then 0-fill.
        arr = pd.DataFrame(arr).ffill().fillna(0).to_numpy(dtype=float)
        seqs[pid] = arr[-max_len:]
    return seqs


def threshold_for_recall(y_true, y_prob, target_recall: float = 0.90) -> float:
    """Lowest threshold achieving at least `target_recall` on the given set."""
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # recalls is decreasing with threshold; find first threshold meeting target.
    for r, t in zip(recalls[:-1], thresholds):
        if r >= target_recall:
            return float(t)
    return 0.5
