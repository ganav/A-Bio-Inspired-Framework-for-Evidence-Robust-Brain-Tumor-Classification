#!/usr/bin/env python3
"""
Compare repeated baseline and proposed runs using actual macro-F1 values.

Input CSV format:
run,baseline_f1,proposed_f1
1,0.896,0.928
2,...
...

Outputs:
- paired t-test p-value
- paired Cohen's dz
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_file")
    args = p.parse_args()

    df = pd.read_csv(args.csv_file)
    required = {"baseline_f1", "proposed_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    baseline = df["baseline_f1"].to_numpy(dtype=float)
    proposed = df["proposed_f1"].to_numpy(dtype=float)

    if len(baseline) < 2:
        raise ValueError("At least two paired runs are required.")

    diff = proposed - baseline
    t_stat, p_value = ttest_rel(proposed, baseline)
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd > 0 else float("inf")

    print(f"n paired runs: {len(diff)}")
    print(f"baseline mean F1: {baseline.mean():.6f}")
    print(f"proposed mean F1: {proposed.mean():.6f}")
    print(f"mean difference: {diff.mean():.6f}")
    print(f"paired t statistic: {t_stat:.6f}")
    print(f"p-value: {p_value:.6g}")
    print(f"Cohen\'s dz: {dz:.6f}")


if __name__ == "__main__":
    main()
