

"""Shared pairwise significance testing utilities for experiment scripts.

Provides paired t-test, bootstrap CI, and Cohen's d for comparing
"ours" methods vs baselines across paired seed observations.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats as scipy_stats


def compute_pairwise_significance(
    method_seed_values: dict[str, list[float]],
    ours_methods: set[str],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    higher_is_better: bool = True,
) -> dict[str, object]:
    """Paired t-test + bootstrap CI + Cohen's d for best-ours vs best-baseline.

    Args:
        method_seed_values: {method_name: [value_per_seed_0, value_per_seed_1, ...]}.
            All lists must have the same length (paired observations).
        ours_methods: Set of method names considered "ours" (RQ-based).
        n_bootstrap: Number of bootstrap resamples for CI.
        ci_level: Confidence level for bootstrap CI (e.g. 0.95 for 95%).
        higher_is_better: If True, "best" = highest mean. If False (regret),
            "best" = lowest mean.

    Returns:
        Dict with keys: best_ours, best_baseline, n_seeds, mean_diff, std_diff,
        t_stat, p_value, ci_lo, ci_hi, effect_size_cohens_d, ours_mean, base_mean.
        Empty dict if fewer than 2 paired observations or no ours/baseline methods.
    """
    ours_present = {m for m in method_seed_values if m in ours_methods}
    base_present = {m for m in method_seed_values if m not in ours_methods}

    if not ours_present or not base_present:
        return {}

    def _representative(method: str) -> float:
        return float(np.mean(method_seed_values[method]))

    if higher_is_better:
        best_ours = max(ours_present, key=_representative)
        best_base = max(base_present, key=_representative)
    else:
        best_ours = min(ours_present, key=_representative)
        best_base = min(base_present, key=_representative)

    ours_vals = np.array(method_seed_values[best_ours])
    base_vals = np.array(method_seed_values[best_base])

    n = len(ours_vals)
    if n < 2:
        return {}

    # Difference direction: ours - baseline.
    # For higher_is_better, positive diff means ours is better.
    # For lower_is_better (regret), we flip so positive diff still means ours is better.
    if higher_is_better:
        diffs = ours_vals - base_vals
    else:
        diffs = base_vals - ours_vals  # positive = ours has lower regret = better

    # Paired t-test (on raw values, not flipped)
    t_stat, p_value = scipy_stats.ttest_rel(ours_vals, base_vals)

    # Bootstrap percentile CI on mean difference
    rng = np.random.RandomState(42)
    boot_diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_diffs.append(float(diffs[idx].mean()))
    boot_arr = np.array(boot_diffs)
    alpha_half = (1 - ci_level) / 2
    ci_lo = float(np.percentile(boot_arr, 100 * alpha_half))
    ci_hi = float(np.percentile(boot_arr, 100 * (1 - alpha_half)))

    # Effect size (Cohen's d for paired samples)
    effect_size = float(diffs.mean() / max(diffs.std(ddof=1), 1e-10))

    return {
        "best_ours": best_ours,
        "best_baseline": best_base,
        "n_seeds": n,
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "effect_size_cohens_d": effect_size,
        "ours_mean": float(ours_vals.mean()),
        "base_mean": float(base_vals.mean()),
    }


def log_significance(
    sig_results: dict[str, object],
    logger: logging.Logger,
    metric_name: str = "accuracy",
) -> None:
    """Log significance results in a formatted block.

    Args:
        sig_results: Output of compute_pairwise_significance().
        logger: Logger instance.
        metric_name: Name of the metric for display (e.g. "accuracy", "final regret").
    """
    if not sig_results:
        return

    best_ours = sig_results["best_ours"]
    best_base = sig_results["best_baseline"]
    n_seeds = sig_results["n_seeds"]
    ours_mean = sig_results["ours_mean"]
    base_mean = sig_results["base_mean"]
    mean_diff = sig_results["mean_diff"]
    t_stat = sig_results["t_stat"]
    p_value = sig_results["p_value"]
    ci_lo = sig_results["ci_lo"]
    ci_hi = sig_results["ci_hi"]
    cohens_d = sig_results["effect_size_cohens_d"]

    logger.info(f"  {metric_name}: {best_ours} vs {best_base} ({n_seeds} seeds)")
    logger.info(
        f"    Ours: {ours_mean:.4f}, Base: {base_mean:.4f}, Diff: {mean_diff:+.4f}"
    )
    logger.info(f"    Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    logger.info(f"    Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    logger.info(f"    Effect size (Cohen's d): {cohens_d:.3f}")
    sig_str = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
    logger.info(f"    Significant: {sig_str}")
