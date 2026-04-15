

"""P1 Experiment: D_cut Sweep Ablation.

Shows the optimal trunk depth D_cut and that the MI decay profile
(Fisher Information Decay) predicts it.

Design:
    1. Generate synthetic data with class-correlated RQ codes
       (early levels informative, later levels noisy).
    2. For each D_cut ∈ {1, 2, 3, 4, 6, 8}:
       - Use codes[:, :D_cut] as trunk context for TS.
       - Run bandit for N rounds, record cumulative regret.
       - Compute MI at each RQ level via the plug-in estimator.
    3. Repeat across seeds, report mean ± std.
    4. Output: D_cut-vs-regret table, MI decay curve, identified knee.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from modules.bandits import HierarchicalThompsonSampling


@dataclass
class DcutAblationConfig:
    """Configuration for the D_cut ablation experiment."""

    n_arms: int = 10
    n_rounds: int = 5000
    n_seeds: int = 5
    n_classes: int = 10
    n_levels: int = 8
    n_codes: int = 256
    n_samples: int = 5000  # Pool of synthetic data points
    class_separation: float = 0.7
    dcut_values: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    device: str = "cpu"


# -----------------------------------------------------------------------
# Synthetic data with MI structure
# -----------------------------------------------------------------------


def generate_rq_codes_with_mi_structure(
    n_samples: int,
    n_classes: int,
    n_levels: int,
    n_codes: int,
    class_separation: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic RQ codes with class-correlated structure.

    Early levels have strong MI with the class label; later levels are
    progressively noisier, simulating the trunk-tail information decay.

    Each level uses a small effective alphabet (``eff_k``) so that the
    plug-in MI estimator has enough samples per cell.  With probability
    ``class_weight`` (decaying with level) the code is a deterministic
    function of the class label; otherwise it is uniform-random over
    ``{0, …, eff_k-1}``.  Different levels use different hash-like
    mappings of the class identity so that jointly multiple levels can
    discriminate all classes.

    Args:
        n_samples: Number of samples.
        n_classes: Number of classes.
        n_levels: Total RQ levels (D).
        n_codes: Codebook size per level (K), used as upper bound.
        class_separation: Strength of class signal (0=none, 1=perfect).
        seed: Random seed.

    Returns:
        labels: (n_samples,) int array of class labels.
        codes: (n_samples, n_levels) int array of RQ codes.
    """
    rng = np.random.RandomState(seed)

    labels = rng.randint(0, n_classes, n_samples)
    codes = np.zeros((n_samples, n_levels), dtype=np.int64)

    # Small effective alphabet per level — keeps MI estimation well-conditioned.
    # We use ceil(n_classes^(1/4)) so that ~4 levels are needed to jointly
    # identify a class, producing a richer MI decay curve.
    eff_k = max(2, int(math.ceil(n_classes**0.25)))

    # Pre-compute a random class→code mapping per level so that
    # different levels hash the class differently.
    class_to_code = rng.randint(0, eff_k, size=(n_levels, n_classes))

    for level in range(n_levels):
        decay = math.exp(-level / 2.0)  # exponential MI decay
        class_weight = class_separation * decay

        for i, label in enumerate(labels):
            if rng.random() < class_weight:
                codes[i, level] = class_to_code[level, label]
            else:
                codes[i, level] = rng.randint(0, eff_k)

    return labels, codes


# -----------------------------------------------------------------------
# Plug-in MI estimator (inlined to avoid circular deps with mi_diagnostic)
# -----------------------------------------------------------------------


def _entropy_plugin(labels: np.ndarray, alpha: float = 1.0) -> float:
    """H(Y) with add-α smoothing."""
    counts = Counter(labels.tolist())
    n = len(labels)
    n_classes = len(counts)
    total = n + alpha * n_classes
    h = 0.0
    for c in counts.values():
        p = (c + alpha) / total
        if p > 0:
            h -= p * math.log(p)
    return h


def _conditional_entropy_plugin(
    labels: np.ndarray, codes: np.ndarray, alpha: float = 1.0
) -> float:
    """H(Y | Z) with add-α smoothing."""
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)

    code_keys = [tuple(int(c) for c in row) for row in codes]
    n = len(labels)

    joint_counts: dict[tuple[tuple[int, ...], int], int] = {}
    code_counts: dict[tuple[int, ...], int] = {}

    for ck, lab in zip(code_keys, labels.tolist()):
        joint_counts[(ck, lab)] = joint_counts.get((ck, lab), 0) + 1
        code_counts[ck] = code_counts.get(ck, 0) + 1

    unique_labels = set(labels.tolist())
    n_labels = len(unique_labels)

    cond_h = 0.0
    for ck, cc in code_counts.items():
        p_z = cc / n
        h_y_z = 0.0
        total_s = cc + alpha * n_labels
        for lab in unique_labels:
            jc = joint_counts.get((ck, lab), 0)
            p = (jc + alpha) / total_s
            if p > 0:
                h_y_z -= p * math.log(p)
        cond_h += p_z * h_y_z

    return cond_h


def compute_mi_per_level(
    labels: np.ndarray,
    codes: np.ndarray,
    n_levels: int,
    alpha: float = 1.0,
) -> list[float]:
    """Compute MI at each RQ level using the plug-in estimator.

    I(Y; k_ℓ | Z_{1:ℓ-1}) = H(Y | Z_{1:ℓ-1}) − H(Y | Z_{1:ℓ})

    Returns:
        List of MI values, one per level.
    """
    mis: list[float] = []
    for level in range(n_levels):
        if level == 0:
            h_prev = _entropy_plugin(labels, alpha)
        else:
            h_prev = _conditional_entropy_plugin(labels, codes[:, :level], alpha)
        h_curr = _conditional_entropy_plugin(labels, codes[:, : level + 1], alpha)
        mis.append(max(0.0, h_prev - h_curr))
    return mis


# -----------------------------------------------------------------------
# Bandit runner for a given D_cut
# -----------------------------------------------------------------------


def run_bandit_for_dcut(
    labels: np.ndarray,
    codes: np.ndarray,
    best_arms: np.ndarray,
    n_arms: int,
    n_rounds: int,
    d_cut: int,
    n_codes: int,
    seed: int,
) -> float:
    """Run TS bandit using the first d_cut RQ levels as context.

    Args:
        labels: (N,) class labels for reward lookup.
        codes: (N, D) full RQ codes.
        best_arms: (n_classes,) best arm per class.
        n_arms: Number of arms.
        n_rounds: Number of bandit rounds.
        d_cut: Number of trunk levels to use.
        n_codes: Codebook size per level.
        seed: Random seed.

    Returns:
        Final cumulative regret.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(labels)

    trunk_codes = codes[:, :d_cut]
    bandit = HierarchicalThompsonSampling(
        n_arms=n_arms, n_codes=n_codes, d_cut=d_cut, device="cpu"
    )

    regret = 0.0
    best_reward = 0.8

    for _ in range(n_rounds):
        idx = rng.randint(n_samples)
        ctx = torch.from_numpy(trunk_codes[idx : idx + 1])
        arm = bandit.select_arm(ctx).item()

        # Stochastic reward
        label = int(labels[idx])
        mu = 0.8 if arm == best_arms[label] else 0.2
        reward = float(rng.normal(mu, 0.1))

        bandit.update(ctx, torch.tensor([arm]), torch.tensor([reward]))
        regret += best_reward - mu  # expected regret

    return regret


# -----------------------------------------------------------------------
# Ablation driver
# -----------------------------------------------------------------------


def _find_knee(mi_values: list[float]) -> int:
    """Find the knee level where MI drops below 10% of peak.

    The knee is the first level where MI becomes negligible compared to
    the peak, indicating that adding more trunk levels provides
    diminishing returns.
    """
    if len(mi_values) < 2:
        return 0
    peak = max(mi_values)
    if peak <= 0:
        return 0
    threshold = 0.1 * peak
    for i, mi in enumerate(mi_values):
        if mi < threshold:
            return i
    return len(mi_values)


def run_ablation(config: DcutAblationConfig) -> dict[str, object]:
    """Sweep D_cut values and report regret + MI decay.

    Returns:
        Dictionary with regret arrays and MI profile.
    """
    results: dict[str, object] = {
        "dcut_values": list(config.dcut_values),
        "regret_mean": [],
        "regret_std": [],
        "mi_per_level": [],
        "knee": None,
    }

    print("\n" + "=" * 70)
    print("P1 EXPERIMENT: D_cut Sweep Ablation")
    print("=" * 70)
    print(f"Arms={config.n_arms}  Rounds={config.n_rounds}  Seeds={config.n_seeds}")
    print(
        f"Levels={config.n_levels}  Codes={config.n_codes}  "
        f"ClassSep={config.class_separation}"
    )
    print("=" * 70)

    # Use a representative seed to compute MI profile once
    labels_mi, codes_mi = generate_rq_codes_with_mi_structure(
        n_samples=config.n_samples,
        n_classes=config.n_classes,
        n_levels=config.n_levels,
        n_codes=config.n_codes,
        class_separation=config.class_separation,
        seed=42,
    )
    mi_profile = compute_mi_per_level(labels_mi, codes_mi, config.n_levels)
    results["mi_per_level"] = mi_profile

    print("\nMI decay profile (plug-in estimator):")
    for lv, mi_val in enumerate(mi_profile):
        bar = "#" * int(mi_val * 80 / max(max(mi_profile), 1e-9))
        print(f"  Level {lv}: {mi_val:.4f}  {bar}")

    knee = _find_knee(mi_profile)
    results["knee"] = knee
    print(f"\n  Detected knee at level {knee}")

    # Best arm per class (fixed across seeds)
    rng_arms = np.random.RandomState(0)
    best_arms = rng_arms.randint(0, config.n_arms, config.n_classes)

    # Sweep D_cut
    print(f"\n{'D_cut':>6s}  {'Regret':>20s}")
    print("-" * 34)

    for d_cut in config.dcut_values:
        seed_regrets: list[float] = []
        for s in range(config.n_seeds):
            seed = 1000 * s + 42
            labels, codes = generate_rq_codes_with_mi_structure(
                n_samples=config.n_samples,
                n_classes=config.n_classes,
                n_levels=config.n_levels,
                n_codes=config.n_codes,
                class_separation=config.class_separation,
                seed=seed,
            )
            r = run_bandit_for_dcut(
                labels,
                codes,
                best_arms,
                config.n_arms,
                config.n_rounds,
                d_cut,
                config.n_codes,
                seed,
            )
            seed_regrets.append(r)

        r_mean = float(np.mean(seed_regrets))
        r_std = float(np.std(seed_regrets))
        # pyre-ignore[16]: results values are lists
        results["regret_mean"].append(r_mean)
        # pyre-ignore[16]
        results["regret_std"].append(r_std)
        print(f"{d_cut:6d}  {r_mean:8.1f} ± {r_std:6.1f}")

    # Summary
    # pyre-ignore[16]
    best_idx = int(np.argmin(results["regret_mean"]))
    best_dcut = config.dcut_values[best_idx]
    print(f"\n  Best D_cut = {best_dcut}  (MI knee = {knee})")
    if best_dcut == knee:
        print("  ✓ MI knee matches optimal D_cut!")
    else:
        print(f"  MI knee ({knee}) differs from optimal D_cut ({best_dcut})")

    print("-" * 70)
    return results


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="P1: D_cut Sweep Ablation")
    parser.add_argument("--n-arms", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=5000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-levels", type=int, default=8)
    parser.add_argument("--n-codes", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--class-separation", type=float, default=0.7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = DcutAblationConfig(
        n_arms=args.n_arms,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
        n_classes=args.n_classes,
        n_levels=args.n_levels,
        n_codes=args.n_codes,
        n_samples=args.n_samples,
        class_separation=args.class_separation,
        device=args.device,
    )

    run_ablation(config)

    print("\nEXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
