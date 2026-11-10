"""Temporal feature engineering — 50+ features.

All features are computed *only* from events strictly before the
patient's index_date, so there is no label leakage. Windows are
30 / 90 / 180 days before index.

Feature groups
--------------
- demographics / comorbidity
- utilization: counts by type and window, recency
- labs (per analyte): last, mean, std (volatility), min, max,
  slope, 90-day change, abnormal count
- medications: burden, changes, recency
- vitals: last, mean, volatility, slope
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAB_NAMES = ["creatinine", "hemoglobin", "bnp", "crp", "glucose"]
# Rough "abnormal" thresholds for the synthetic labs (illustrative only).
ABNORMAL_HIGH = {"creatinine": 1.3, "bnp": 300.0, "crp": 20.0, "glucose": 180.0}
ABNORMAL_LOW = {"hemoglobin": 11.0}


def _slope(days_before: np.ndarray, values: np.ndarray) -> float:
    """Slope of value vs. time (per day, index = 0). Returns 0 if too few points."""
    if len(values) < 2 or np.std(days_before) == 0:
        return 0.0
    # days_before decreases toward index; use -days so slope is "per day toward index".
    return float(np.polyfit(-days_before.astype(float), values.astype(float), 1)[0])


def _window_stats(df: pd.DataFrame, value_col: str = "lab_value") -> dict:
    if df.empty:
        return {"last": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "slope": 0.0, "change_90d": 0.0, "n": 0, "abnormal": 0}
    vals = df[value_col].to_numpy(dtype=float)
    days = df["days_before_index"].to_numpy(dtype=float)
    # Last = closest to index (smallest days_before).
    last = float(df.sort_values("days_before_index").iloc[0][value_col])
    recent = df[df["days_before_index"] <= 90]
    older = df[df["days_before_index"] > 90]
    change_90d = 0.0
    if not recent.empty and not older.empty:
        change_90d = float(recent[value_col].mean() - older[value_col].mean())
    return {
        "last": last,
        "mean": float(vals.mean()),
        "std": float(vals.std()) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "slope": _slope(days, vals),
        "change_90d": change_90d,
        "n": int(len(vals)),
        "abnormal": 0,  # filled in by caller per-lab
    }


def build_features(data) -> pd.DataFrame:
    """Build one row per patient. `data` is a SyntheticData object."""
    patients = data.patients.copy()
    patients["index_date"] = pd.to_datetime(patients["index_date"])

    enc = data.encounters.copy()
    enc["event_date"] = pd.to_datetime(enc["event_date"])
    labs = data.labs.copy()
    labs["event_date"] = pd.to_datetime(labs["event_date"])
    meds = data.medications.copy()
    meds["event_date"] = pd.to_datetime(meds["event_date"])
    vitals = data.vitals.copy()
    vitals["event_date"] = pd.to_datetime(vitals["event_date"])

    # Pre-merge index dates for days_before calculations.
    idx_map = patients.set_index("patient_id")["index_date"]

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["index_date"] = df["patient_id"].map(idx_map)
        df["days_before_index"] = (df["index_date"] - df["event_date"]).dt.days
        # Keep only pre-index events.
        return df[df["days_before_index"] > 0]

    enc, labs, meds, vitals = _prep(enc), _prep(labs), _prep(meds), _prep(vitals)

    rows = []
    for _, p in patients.iterrows():
        pid = p["patient_id"]
        f: dict = {
            "patient_id": pid,
            "age": float(p["age"]),
            "sex_M": 1.0 if p["sex"] == "M" else 0.0,
            "comorbidity_count": float(p["comorbidity_count"]),
            "age_x_comorbidity": float(p["age"] * p["comorbidity_count"]),
            "index_date": p["index_date"],
            "hospitalized_30d": int(p["hospitalized_30d"]),
            "hospitalized_90d": int(p["hospitalized_90d"]),
        }

        # --- Utilization ---
        e = enc[enc["patient_id"] == pid]
        for window in (30, 90, 180):
            w = e[e["days_before_index"] <= window]
            f[f"encounters_{window}d"] = float(len(w))
            f[f"ed_{window}d"] = float((w["encounter_type"] == "ED").sum())
            f[f"inpatient_{window}d"] = float((w["encounter_type"] == "inpatient").sum())
            f[f"outpatient_{window}d"] = float((w["encounter_type"] == "outpatient").sum())
        f["days_since_last_encounter"] = float(e["days_before_index"].min()) if not e.empty else 180.0
        f["days_since_last_ed"] = (
            float(e[e["encounter_type"] == "ED"]["days_before_index"].min())
            if (e["encounter_type"] == "ED").any() else 180.0
        )
        f["utilization_trend_90d_vs_180d"] = (
            f["encounters_90d"] - (f["encounters_180d"] - f["encounters_90d"])
        )

        # --- Labs: 9 stats x 5 analytes = 45 features ---
        l = labs[labs["patient_id"] == pid]
        for lab in LAB_NAMES:
            sub = l[l["lab_name"] == lab]
            stats = _window_stats(sub)
            prefix = lab
            f[f"{prefix}_last"] = stats["last"]
            f[f"{prefix}_mean"] = stats["mean"]
            f[f"{prefix}_std"] = stats["std"]
            f[f"{prefix}_slope"] = stats["slope"]
            f[f"{prefix}_change_90d"] = stats["change_90d"]
            f[f"{prefix}_range"] = stats["max"] - stats["min"]
            f[f"{prefix}_n"] = float(stats["n"])
            if lab in ABNORMAL_HIGH:
                f[f"{prefix}_abnormal_n"] = float((sub["lab_value"] > ABNORMAL_HIGH[lab]).sum())
            elif lab in ABNORMAL_LOW:
                f[f"{prefix}_abnormal_n"] = float((sub["lab_value"] < ABNORMAL_LOW[lab]).sum())
            else:
                f[f"{prefix}_abnormal_n"] = 0.0
            # Volatility relative to mean (coefficient of variation).
            f[f"{prefix}_cv"] = stats["std"] / (abs(stats["mean"]) + 1e-6)

        # --- Medications ---
        m = meds[meds["patient_id"] == pid]
        if not m.empty:
            f["meds_last_n"] = float(m.sort_values("days_before_index").iloc[0]["n_medications"])
            f["meds_mean_n"] = float(m["n_medications"].mean())
            f["meds_max_n"] = float(m["n_medications"].max())
            f["med_changes_180d"] = float(m["medication_change"].sum())
            f["med_changes_90d"] = float(m[m["days_before_index"] <= 90]["medication_change"].sum())
            f["days_since_last_med_change"] = (
                float(m[m["medication_change"] == 1]["days_before_index"].min())
                if (m["medication_change"] == 1).any() else 180.0
            )
        else:
            f.update({k: 0.0 for k in
                      ["meds_last_n", "meds_mean_n", "meds_max_n",
                       "med_changes_180d", "med_changes_90d"]})
            f["days_since_last_med_change"] = 180.0

        # --- Vitals ---
        v = vitals[vitals["patient_id"] == pid]
        for col, prefix in (("systolic_bp", "sbp"), ("heart_rate", "hr")):
            if v.empty:
                f[f"{prefix}_last"] = 0.0
                f[f"{prefix}_mean"] = 0.0
                f[f"{prefix}_std"] = 0.0
                f[f"{prefix}_slope"] = 0.0
            else:
                sub = v[["days_before_index", col]].rename(columns={col: "lab_value"})
                stats = _window_stats(sub)
                f[f"{prefix}_last"] = stats["last"]
                f[f"{prefix}_mean"] = stats["mean"]
                f[f"{prefix}_std"] = stats["std"]
                f[f"{prefix}_slope"] = stats["slope"]

        rows.append(f)

    out = pd.DataFrame(rows)
    return out


FEATURE_COLUMNS = None  # populated by get_feature_columns()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"patient_id", "index_date", "hospitalized_30d", "hospitalized_90d"}
    return [c for c in df.columns if c not in exclude]
