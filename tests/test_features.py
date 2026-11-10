import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpip.data import generate_synthetic_data
from lpip.features import build_features, get_feature_columns


def test_features_count_and_no_leakage_columns():
    data = generate_synthetic_data(n_patients=80, seed=2)
    feats = build_features(data)
    cols = get_feature_columns(feats)
    # 50+ temporal features is the design goal
    assert len(cols) >= 50, f"only {len(cols)} features: {cols}"
    assert len(feats) == 80
    assert feats[cols].isna().sum().sum() == 0
    assert "hospitalized_30d" not in cols
