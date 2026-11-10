"""Evaluation: discrimination, calibration, and subgroup checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def subgroup_metrics(df: pd.DataFrame, y_true, y_prob, group_col: str) -> dict:
    """df must be row-aligned with y_true / y_prob (reset index first)."""
    out = {}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for name in sorted(df[group_col].astype(str).unique()):
        mask = (df[group_col].astype(str) == str(name)).to_numpy()
        if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
            continue
        out[str(name)] = classification_metrics(y_true[mask], y_prob[mask])
    return out


def add_subgroup_columns(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 50, 65, 80, 200], labels=["<50", "50-64", "65-79", "80+"]
    ).astype(str)
    df["sex_group"] = np.where(df["sex_M"] == 1, "M", "F")
    df["comorbidity_group"] = np.where(
        df["comorbidity_count"] >= 3, "3+ comorbidities", "0-2 comorbidities"
    )
    return df
