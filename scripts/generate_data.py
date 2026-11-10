"""Generate synthetic data CSVs. Usage: python scripts/generate_data.py --n-patients 2000 --out-dir data"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpip.data import generate_synthetic_data, save_synthetic_data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-patients", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="data")
    args = p.parse_args()
    save_synthetic_data(generate_synthetic_data(args.n_patients, args.seed), args.out_dir)
    print(f"Wrote synthetic data to {args.out_dir}/")
