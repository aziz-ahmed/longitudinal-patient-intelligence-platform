"""End-to-end pipeline: generate -> features -> temporal split -> train -> evaluate.

Usage:
    python -m lpip.pipeline --n-patients 2000 --seed 42 --output-dir outputs

Temporal split: train on index dates before the cutoff, test on/after it.
No random split — this mimics prospective deployment.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from .data import generate_synthetic_data, save_synthetic_data
from .evaluate import add_subgroup_columns, classification_metrics, subgroup_metrics
from .explain import explain_model
from .features import build_features, get_feature_columns
from .models import build_models


def temporal_split(features: pd.DataFrame, cutoff: pd.Timestamp | None = None):
    if cutoff is None:
        cutoff = features["index_date"].median()
    train = features[features["index_date"] < cutoff].reset_index(drop=True)
    test = features[features["index_date"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def plot_figures(y_true, probs_by_model: dict, out_dir: str, horizon: str):
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    fig, ax = plt.subplots()
    for name, prob in probs_by_model.items():
        RocCurveDisplay.from_predictions(y_true, prob, name=name, ax=ax)
    ax.set_title(f"ROC — {horizon}")
    fig.savefig(f"{out_dir}/figures/roc_{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for name, prob in probs_by_model.items():
        PrecisionRecallDisplay.from_predictions(y_true, prob, name=name, ax=ax)
    ax.set_title(f"Precision-Recall — {horizon}")
    fig.savefig(f"{out_dir}/figures/pr_{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(n_patients: int = 2000, seed: int = 42, output_dir: str = "outputs",
        save_data: bool = True) -> dict:
    print(f"Generating synthetic data: n_patients={n_patients}, seed={seed}")
    data = generate_synthetic_data(n_patients=n_patients, seed=seed)
    if save_data:
        save_synthetic_data(data, os.path.join(output_dir, "data"))

    print("Building temporal features...")
    features = build_features(data)
    feature_cols = get_feature_columns(features)
    print(f"  {len(feature_cols)} features, {len(features)} patients")

    train, test, cutoff = temporal_split(features)
    print(f"Temporal split at {cutoff.date()}: train={len(train)}, test={len(test)}")

    models = build_models(random_state=seed)
    results: dict = {
        "n_patients": int(n_patients),
        "seed": int(seed),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "temporal_cutoff": str(cutoff.date()),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
    }

    test_with_groups = add_subgroup_columns(test)

    for horizon, label_col in (("30d", "hospitalized_30d"), ("90d", "hospitalized_90d")):
        print(f"\nHorizon: {horizon} ({label_col})")
        y_train = train[label_col].to_numpy()
        y_test = test[label_col].to_numpy()
        probs: dict[str, np.ndarray] = {}
        results[f"horizon_{horizon}"] = {}
        for name, model in models.items():
            model.fit(train[feature_cols], y_train)
            prob = model.predict_proba(test[feature_cols])[:, 1]
            probs[name] = prob
            metrics = classification_metrics(y_test, prob)
            results[f"horizon_{horizon}"][name] = metrics
            print(f"  {name:22s} AUROC={metrics['auroc']:.3f}  "
                  f"AUPRC={metrics['auprc']:.3f}  recall={metrics['recall_sensitivity']:.3f}  "
                  f"precision={metrics['precision']:.3f}  Brier={metrics['brier']:.3f}")

            # Subgroup checks for the primary model only, to keep output readable.
            if name == "xgboost" or (name == "hist_gb" and "xgboost" not in models):
                for group_col in ("age_group", "sex_group", "comorbidity_group"):
                    sg = subgroup_metrics(test_with_groups, y_test, prob, group_col)
                    results[f"horizon_{horizon}"][f"subgroups_{group_col}"] = sg

        plot_figures(y_test, probs, output_dir, horizon)

        # Explainability on the primary model.
        primary_name = "xgboost" if "xgboost" in models else "hist_gb"
        primary = models[primary_name]
        # Refit is already done above; explain on test sample.
        try:
            importance, method = explain_model(primary, test[feature_cols], y_test)
            results[f"horizon_{horizon}"]["explainability"] = {
                "method": method,
                "top_features": importance.head(15).to_dict(orient="records"),
            }
            print(f"  Explainability ({method}): "
                  f"{', '.join(importance.head(5)['feature'])}")
            importance.head(20).to_csv(
                f"{output_dir}/feature_importance_{horizon}.csv", index=False
            )
        except Exception as exc:  # explainability must not break the pipeline
            print(f"  Explainability skipped: {exc}")
            results[f"horizon_{horizon}"]["explainability"] = {"method": "skipped", "error": str(exc)}

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/metrics.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nWrote {output_dir}/metrics.json")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--no-save-data", action="store_true")
    args = parser.parse_args()
    run(
        n_patients=args.n_patients,
        seed=args.seed,
        output_dir=args.output_dir,
        save_data=not args.no_save_data,
    )


if __name__ == "__main__":
    main()
