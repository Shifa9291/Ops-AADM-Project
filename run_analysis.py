#!/usr/bin/env python
"""Command-line entry point.

    python run_analysis.py                       # uses data/OPS_DS1.csv, writes to outputs/
    python run_analysis.py --data path/to.csv --out results --fast
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aadm.pipeline import run_pipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="AADM end-term project - supply chain analytics")
    p.add_argument("--data", default="data/OPS_DS1.csv", help="path to OPS_DS1.csv / .xlsx")
    p.add_argument("--out", default="outputs", help="folder for figures, results.txt, metrics.json")
    p.add_argument("--fast", action="store_true",
                   help="smaller random-forest grid (quick smoke test, results will differ)")
    a = p.parse_args()
    run_pipeline(a.data, a.out, fast=a.fast)


if __name__ == "__main__":
    main()
