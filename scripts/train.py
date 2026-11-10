"""Train + evaluate. Usage: python scripts/train.py --n-patients 2000 --output-dir outputs"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpip.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-patients", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs")
    args = p.parse_args()
    run(n_patients=args.n_patients, seed=args.seed, output_dir=args.output_dir)
