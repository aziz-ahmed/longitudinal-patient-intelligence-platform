"""Synthetic longitudinal data generator.

Generates EHR-like event tables from scratch with a seeded RNG.
No real patient data, no PHI, no client data.

Tables
------
patients:    patient_id, age, sex, comorbidity_count, index_date,
             hospitalized_30d, hospitalized_90d
encounters:  patient_id, event_date, encounter_type (outpatient/ED/inpatient)
labs:        patient_id, event_date, lab_name, lab_value
medications: patient_id, event_date, n_medications, medication_change (0/1)
vitals:      patient_id, event_date, systolic_bp, heart_rate

Labels are generated from a latent risk score so the signal is learnable
but not deterministic. Deteriorating patients have worsening lab trends
and more recent utilization — the temporal patterns the feature layer
is designed to capture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABS = {
    # name: (typical mean, typical sd, high-is-bad?)
    "creatinine": (1.0, 0.3, True),
    "hemoglobin": (13.5, 1.5, False),
    "bnp": (120.0, 80.0, True),
    "crp": (8.0, 6.0, True),
    "glucose": (110.0, 25.0, True),
}


@dataclass
class SyntheticData:
    patients: pd.DataFrame
    encounters: pd.DataFrame
    labs: pd.DataFrame
    medications: pd.DataFrame
    vitals: pd.DataFrame


def generate_synthetic_data(
    n_patients: int = 2000,
    seed: int = 42,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> SyntheticData:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    patient_ids = [f"SYN-{i:05d}" for i in range(1, n_patients + 1)]
    age = rng.normal(68, 14, n_patients).clip(18, 95).round().astype(int)
    sex = rng.choice(["F", "M"], size=n_patients, p=[0.52, 0.48])
    comorbidity = rng.poisson(2.2, n_patients).clip(0, 8)
    frailty = rng.normal(0, 1, n_patients)

    # Latent deterioration: some patients are on a worsening trajectory.
    deteriorating = rng.random(n_patients) < 0.35
    deterioration_rate = np.where(
        deteriorating, rng.normal(0.6, 0.3, n_patients), rng.normal(0.0, 0.15, n_patients)
    )

    # Latent risk drives both the longitudinal patterns and the labels.
    # Frailty is deliberately kept small: most of the risk should be
    # observable in the longitudinal history (labs, utilization), otherwise
    # no model could learn it from the features.
    latent = (
        0.04 * (age - 65)
        + 0.45 * comorbidity
        + 0.20 * frailty
        + 1.4 * deterioration_rate
    )
    p30 = 1 / (1 + np.exp(-(-2.6 + latent)))
    p90 = 1 / (1 + np.exp(-(-1.9 + latent)))
    y30 = (rng.random(n_patients) < p30).astype(int)
    y90 = np.maximum(y30, (rng.random(n_patients) < p90).astype(int))

    # Index dates spread across the window, so a temporal split is meaningful.
    index_dates = start + pd.to_timedelta(
        rng.integers(180, (end - start).days, size=n_patients), unit="D"
    )

    patients = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": age,
            "sex": sex,
            "comorbidity_count": comorbidity,
            "index_date": index_dates,
            "hospitalized_30d": y30,
            "hospitalized_90d": y90,
        }
    )

    encounters, labs, medications, vitals = [], [], [], []

    for i, pid in enumerate(patient_ids):
        idx = index_dates[i]
        # Sicker patients use care more often, especially recently.
        base_rate = 0.02 + 0.012 * comorbidity[i] + 0.05 * max(0, deterioration_rate[i])
        n_enc = rng.poisson(180 * base_rate)
        for _ in range(n_enc):
            days_before = int(rng.integers(1, 181))
            # Recency bias: deteriorating patients cluster in the last 30 days.
            if deteriorating[i] and rng.random() < 0.4:
                days_before = int(rng.integers(1, 31))
            enc_date = idx - pd.Timedelta(days=days_before)
            enc_type = rng.choice(
                ["outpatient", "ED", "inpatient"], p=[0.72, 0.20, 0.08]
            )
            encounters.append((pid, enc_date, enc_type))

        # Labs: repeated measures with a patient-specific slope.
        for lab_name, (mean, sd, high_bad) in LABS.items():
            n_lab = int(rng.integers(4, 13))
            days = np.sort(rng.integers(1, 181, size=n_lab))
            slope = deterioration_rate[i] * (sd * 0.7 if high_bad else -sd * 0.5)
            for d in days:
                # Value drifts as index approaches for deteriorating patients.
                # Stronger, clearer trend so the temporal features can capture it.
                trend = slope * (1 - d / 180.0) * 5
                value = rng.normal(mean + trend, sd * 0.4)
                labs.append((pid, idx - pd.Timedelta(days=int(d)), lab_name, max(0.1, value)))

        # Medications: snapshots, more meds / changes for sicker patients.
        n_med_events = int(rng.integers(2, 7))
        for _ in range(n_med_events):
            d = int(rng.integers(1, 181))
            n_meds = int(np.clip(rng.poisson(3 + comorbidity[i] * 0.7), 0, 15))
            change = int(rng.random() < (0.25 + 0.3 * max(0, deterioration_rate[i])))
            medications.append((pid, idx - pd.Timedelta(days=d), n_meds, change))

        # Vitals: repeated measures.
        n_vit = int(rng.integers(4, 11))
        for _ in range(n_vit):
            d = int(rng.integers(1, 181))
            sbp = rng.normal(128 + 6 * deterioration_rate[i], 14)
            hr = rng.normal(76 + 5 * deterioration_rate[i], 10)
            vitals.append((pid, idx - pd.Timedelta(days=d), sbp, hr))

    encounters_df = pd.DataFrame(
        encounters, columns=["patient_id", "event_date", "encounter_type"]
    )
    labs_df = pd.DataFrame(
        labs, columns=["patient_id", "event_date", "lab_name", "lab_value"]
    )
    medications_df = pd.DataFrame(
        medications,
        columns=["patient_id", "event_date", "n_medications", "medication_change"],
    )
    vitals_df = pd.DataFrame(
        vitals, columns=["patient_id", "event_date", "systolic_bp", "heart_rate"]
    )

    return SyntheticData(
        patients=patients,
        encounters=encounters_df,
        labs=labs_df,
        medications=medications_df,
        vitals=vitals_df,
    )


def save_synthetic_data(data: SyntheticData, out_dir: str) -> None:
    """Save all synthetic tables as CSVs (no PHI — IDs are SYN-xxxxx)."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    data.patients.to_csv(f"{out_dir}/patients.csv", index=False)
    data.encounters.to_csv(f"{out_dir}/encounters.csv", index=False)
    data.labs.to_csv(f"{out_dir}/labs.csv", index=False)
    data.medications.to_csv(f"{out_dir}/medications.csv", index=False)
    data.vitals.to_csv(f"{out_dir}/vitals.csv", index=False)
