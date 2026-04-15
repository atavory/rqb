#!/usr/bin/env python3
"""Compute significance tests for bandit experiment results.

Reads the test_runner_results.md tables (or raw JSON) and computes:
- Mean ± std per method per depth
- Paired t-tests (RQ vs each baseline)
- Bootstrap confidence intervals
- Cohen's d effect sizes

Usage:
    python3 scripts/significance_tests.py
    python3 scripts/significance_tests.py --input results.csv --alpha 0.05
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def parse_covertype_table(results_path: str) -> dict[str, dict[int, list[float]]]:
    """Parse the Covertype full results table from test_runner_results.md.

    Returns {method: {depth: [regret_per_seed]}}.
    """
    text = Path(results_path).read_text()

    # Find the covertype table (lines with | d | seed | RQ-LinTS | ...)
    methods = [
        "RQ-LinTS", "ProjRQ-16", "ProjRQ-8", "TS", "LinTS",
        "PCA-8", "PCA-16", "RP-16", "RP-8", "KM-LinTS", "Random"
    ]
    data = {m: {} for m in methods}

    # Match lines like: | 1 | 42 | 69.4 | 109.2 | ... |
    pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        + r"\s*([\d,.]+)\s*\|" * len(methods)
    )

    for line in text.split("\n"):
        m = pattern.search(line)
        if m:
            d = int(m.group(1))
            # seed = int(m.group(2))
            for i, method in enumerate(methods):
                val_str = m.group(3 + i).replace(",", "")
                val = float(val_str)
                if d not in data[method]:
                    data[method][d] = []
                data[method][d].append(val)

    return data


def parse_letter_table(results_path: str) -> dict[str, dict[int, list[float]]]:
    """Parse Letter results similarly."""
    # Same structure, different section
    return parse_covertype_table(results_path)


def load_manual_data() -> dict[str, dict[str, dict[int, list[float]]]]:
    """Load hardcoded results from the session.

    Returns {dataset: {method: {depth: [regret_per_seed]}}}.
    """
    datasets = {}

    # Covertype 25/25 complete (from test_runner_results.md)
    datasets["covertype"] = {
        "RQ-LinTS": {
            1: [69.4, 78.9, 72.0, 116.0, 81.8],
            2: [109.4, 162.0, 115.1, 111.5, 87.1],
            3: [907.0, 754.2, 799.8, 903.8, 722.9],
            4: [2396.7, 2365.5, 2493.3, 2438.9, 2451.4],
            5: [2358.1, 2372.7, 2487.1, 2375.7, 2555.6],
        },
        "ProjRQ-16": {
            1: [109.2, 47.4, 76.6, 90.1, 143.4],
            2: [93.0, 116.2, 179.7, 64.8, 319.8],
            3: [804.5, 870.3, 925.2, 820.5, 1199.7],
            4: [2510.7, 2442.6, 2403.4, 2494.5, 2351.6],
            5: [2285.4, 2413.5, 2372.1, 2537.3, 2588.7],
        },
        "PCA-8": {
            1: [40.5, 80.3, 27.2, 25.5, 46.0],
            2: [1610.6, 1670.1, 1680.1, 1738.1, 1701.1],
            3: [4595.3, 4449.4, 4251.7, 4389.9, 4530.0],
            4: [3921.6, 3959.0, 3990.4, 4027.7, 4014.4],
            5: [3842.7, 3918.5, 3892.5, 3913.4, 3948.8],
        },
        "TS": {
            1: [150.6, 162.2, 204.7, 1773.1, 137.4],
            2: [502.2, 483.0, 250.1, 733.7, 990.1],
            3: [716.5, 701.0, 657.9, 631.9, 813.8],
            4: [1934.9, 1955.3, 1804.8, 1692.3, 1880.0],
            5: [3681.4, 3839.0, 3679.5, 3714.6, 3860.5],
        },
        "LinTS": {
            1: [808.0, 703.0, 611.5, 622.9, 794.5],
            2: [1592.7, 1578.9, 1704.9, 1747.9, 1708.5],
            3: [3692.6, 3769.0, 3487.7, 3810.7, 3682.3],
            4: [3637.5, 3492.9, 3514.3, 3870.4, 3632.0],
            5: [2693.4, 3026.4, 2897.7, 2851.5, 3059.5],
        },
    }

    # Adult 10 seeds at d=3 (from test_runner_results.md)
    datasets["adult"] = {
        "RQ-LinTS": {
            3: [221.9, 251.5, 223.0, 203.1, 264.7, 245.4, 263.8, 223.8, 204.8, 227.4],
        },
        "ProjRQ-16": {
            3: [274.1, 310.0, 214.6, 246.4, 278.2, 254.4, 245.1, 223.5, 233.2, 182.8],
        },
        "PCA-8": {
            3: [379.6, 381.7, 378.3, 387.0, 371.8, 369.9, 372.8, 368.1, 341.9, 386.3],
        },
        "LinTS": {
            3: [329.1, 368.7, 327.4, 340.9, 386.6, 349.0, 375.7, 382.1, 359.3, 387.3],
        },
    }

    return datasets


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test. Returns (t_stat, p_value)."""
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 2:
        return 0.0, 1.0
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (paired)."""
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    if np.std(diff) < 1e-10:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, ci_low, ci_high)."""
    arr = np.array(values)
    boot_means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lo), float(hi)


def pct_gap(a_mean: float, b_mean: float) -> float:
    """Percentage gap: how much better a is than b."""
    if b_mean == 0:
        return 0.0
    return 100.0 * (b_mean - a_mean) / b_mean


def analyze_dataset(
    name: str,
    data: dict[str, dict[int, list[float]]],
    reference: str = "RQ-LinTS",
    alpha: float = 0.05,
):
    """Run significance analysis for one dataset."""
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"{'='*70}")

    depths = sorted(set(d for m in data.values() for d in m.keys()))

    for d in depths:
        print(f"\n--- Depth {d} ---")

        # Get reference method data
        if reference not in data or d not in data[reference]:
            print(f"  No {reference} data at depth {d}")
            continue

        ref_vals = data[reference][d]
        ref_mean, ref_lo, ref_hi = bootstrap_ci(ref_vals, alpha=alpha)
        n_ref = len(ref_vals)

        print(f"  {reference}: {ref_mean:.1f} [{ref_lo:.1f}, {ref_hi:.1f}] (n={n_ref})")
        print()
        print(f"  {'Method':<15} {'Mean':>8} {'Std':>8} {'n':>3} {'Gap%':>7} {'t-stat':>7} {'p-val':>8} {'sig':>5} {'Cohen d':>8}")

        methods_at_depth = [(m, data[m][d]) for m in data if d in data[m] and m != reference]
        methods_at_depth.sort(key=lambda x: np.mean(x[1]))

        for method, vals in methods_at_depth:
            n = len(vals)
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if n > 1 else 0
            gap = pct_gap(ref_mean, mean)
            t, p = paired_ttest(ref_vals, vals)
            d_eff = cohens_d(ref_vals, vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else ""

            print(f"  {method:<15} {mean:>8.1f} {std:>8.1f} {n:>3} {gap:>+6.1f}% {t:>7.2f} {p:>8.4f} {sig:>5} {d_eff:>+8.2f}")

    # Summary: best method per depth
    print(f"\n--- Best method per depth ---")
    for d in depths:
        methods_at_depth = [(m, np.mean(data[m][d])) for m in data if d in data[m]]
        methods_at_depth.sort(key=lambda x: x[1])
        best_m, best_v = methods_at_depth[0]
        second_m, second_v = methods_at_depth[1] if len(methods_at_depth) > 1 else ("—", 0)
        gap = pct_gap(best_v, second_v)
        print(f"  d={d}: {best_m} ({best_v:.1f}) > {second_m} ({second_v:.1f}), gap={gap:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Significance tests for bandit results")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--reference", type=str, default="RQ-LinTS", help="Reference method")
    args = parser.parse_args()

    np.random.seed(42)

    datasets = load_manual_data()

    for name, data in datasets.items():
        analyze_dataset(name, data, reference=args.reference, alpha=args.alpha)


if __name__ == "__main__":
    main()
