"""Explainability: SHAP when available, permutation importance as fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def top_features_shap(model, X: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Mean |SHAP| per feature for tree models. Raises if SHAP is unavailable."""
    import shap

    sample = X.sample(n=min(max_rows, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    importance = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_features_permutation(model, X, y, n_repeats: int = 5) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, scoring="roc_auc"
    )
    return (
        pd.DataFrame(
            {"feature": X.columns, "mean_importance": result.importances_mean}
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_model(model, X: pd.DataFrame, y) -> tuple[pd.DataFrame, str]:
    """Return (importance_df, method_used). Prefers SHAP for XGBoost/HistGB."""
    try:
        # Unwrap sklearn Pipeline if present.
        inner = model
        if hasattr(model, "steps"):
            inner = model.steps[-1][1]
        return top_features_shap(inner, X), "shap"
    except Exception:
        return top_features_permutation(model, X, y), "permutation_importance"
