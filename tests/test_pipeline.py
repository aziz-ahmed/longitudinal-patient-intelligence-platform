import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpip.pipeline import run


def test_pipeline_smoke(tmp_path):
    results = run(n_patients=200, seed=3, output_dir=str(tmp_path), save_data=False)
    for horizon in ("horizon_30d", "horizon_90d"):
        assert horizon in results
        # At least one model should beat chance on synthetic data
        aurocs = [m["auroc"] for m in results[horizon].values() if isinstance(m, dict) and "auroc" in m]
        assert max(aurocs) > 0.6, aurocs
    assert (tmp_path / "metrics.json").exists()
