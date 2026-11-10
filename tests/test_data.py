import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpip.data import generate_synthetic_data


def test_generate_shapes():
    data = generate_synthetic_data(n_patients=100, seed=1)
    assert len(data.patients) == 100
    assert set(data.patients.columns) >= {"patient_id", "hospitalized_30d", "hospitalized_90d"}
    assert data.patients["patient_id"].str.startswith("SYN-").all()
    assert not data.encounters.empty
    assert not data.labs.empty
    # 90-day label includes 30-day positives by construction
    assert (data.patients["hospitalized_90d"] >= data.patients["hospitalized_30d"]).all()


def test_reproducible():
    a = generate_synthetic_data(n_patients=50, seed=7)
    b = generate_synthetic_data(n_patients=50, seed=7)
    assert a.patients.equals(b.patients)
