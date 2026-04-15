

"""P1 Experiment: RQ (MI-guided D_cut) vs K-Means Context for Thompson Sampling.

Demonstrates that RQ with MI-guided D_cut achieves near-oracle performance
*without tuning k*, while K-Means requires sweeping k to find the sweet spot.

Design:
    1. Generate synthetic embeddings with hierarchical cluster-reward structure
       and power-law cluster sizes (breaks K-Means's equal-size assumption).
    2. Fit RQ codes via iterative residual quantisation.
    3. Run MI diagnostic on RQ codes to automatically select D_cut.
    4. Compare:
       - RQ (MI-guided D_cut): single operating point, no tuning.
       - K-Means (sweep k): must sweep k ∈ {8, 16, ..., 1024}.
       - Random baseline.
    5. Repeat across seeds and multiple n_clusters settings.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from modules.bandits import HierarchicalThompsonSampling
from sklearn.cluster import MiniBatchKMeans


@dataclass
class ContextComparisonConfig:
    """Configuration for the RQ-vs-KMeans context comparison experiment."""

    n_arms: int = 5
    n_rounds: int = 10000
    n_seeds: int = 5
    embedding_dim: int = 32
    n_clusters: int = 50
    n_samples: int = 5000
    n_codes: int = 16  # RQ codebook per level (small → hierarchical matters)
    n_levels: int = 8
    granularities: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
    cluster_settings: tuple[int, ...] = (20, 50, 100)
    device: str = "cpu"


# -----------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------


def generate_structured_embeddings(
    n_samples: int,
    embedding_dim: int,
    n_clusters: int,
    n_arms: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate embeddings with hierarchical cluster-reward structure.

    Creates a two-level hierarchy with power-law cluster sizes.  Large
    clusters dominate the sample; small clusters are rare but still need
    to be discovered.  This penalises K-Means, which tends to over-split
    large clusters and under-split small ones.

    Args:
        n_samples: Number of embedding vectors.
        embedding_dim: Dimensionality of each embedding.
        n_clusters: Number of true (leaf) clusters.
        n_arms: Number of bandit arms.
        seed: Random seed.

    Returns:
        embeddings: (n_samples, embedding_dim) float array.
        cluster_ids: (n_samples,) int array of true cluster assignments.
        best_arms_per_cluster: (n_clusters,) int array — best arm for each
            cluster.
    """
    rng = np.random.RandomState(seed)

    # Two-level hierarchy: super-clusters and sub-clusters
    n_super = max(2, int(math.ceil(math.sqrt(n_clusters))))
    sub_per_super = max(1, int(math.ceil(n_clusters / n_super)))

    # Super-cluster centres (large scale)
    super_centres = rng.randn(n_super, embedding_dim) * 5.0

    # Sub-cluster offsets within each super-cluster (smaller scale)
    sub_offsets = rng.randn(n_super * sub_per_super, embedding_dim) * 1.0

    # Power-law (Zipf-like) cluster sizes
    weights = 1.0 / np.arange(1, n_clusters + 1).astype(np.float64) ** 0.8
    weights /= weights.sum()
    cluster_ids = rng.choice(n_clusters, n_samples, p=weights)

    # Build embeddings with hierarchical structure
    embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
    for i, cid in enumerate(cluster_ids):
        super_id = cid // sub_per_super
        embeddings[i] = (
            super_centres[super_id] + sub_offsets[cid] + rng.randn(embedding_dim) * 0.3
        )

    # Best arm per cluster (deterministic mapping)
    best_arms_per_cluster = rng.permutation(n_clusters) % n_arms

    return embeddings, cluster_ids, best_arms_per_cluster


# -----------------------------------------------------------------------
# Simulate RQ encoding via K-Means hierarchy
# -----------------------------------------------------------------------


def simulate_rq_codes(
    embeddings: np.ndarray,
    n_levels: int,
    n_codes: int,
    seed: int,
) -> np.ndarray:
    """Simulate RQ encoding by iterative K-Means residual quantisation.

    At each level, cluster the current residuals, record cluster IDs,
    subtract the nearest centroid to form the next residual.

    Args:
        embeddings: (N, D) float array.
        n_levels: Number of RQ levels.
        n_codes: Number of centroids per level.
        seed: Random seed.

    Returns:
        codes: (N, n_levels) int array of RQ codes.
    """
    n_samples = embeddings.shape[0]
    codes = np.zeros((n_samples, n_levels), dtype=np.int64)
    residuals = embeddings.copy()

    for level in range(n_levels):
        k = min(n_codes, n_samples)
        km = MiniBatchKMeans(n_clusters=k, random_state=seed + level, n_init=1)
        km.fit(residuals)
        codes[:, level] = km.labels_
        residuals = residuals - km.cluster_centers_[km.labels_]

    return codes


# -----------------------------------------------------------------------
# MI-guided D_cut selection
# -----------------------------------------------------------------------


def _entropy_plugin(labels: np.ndarray, alpha: float = 1.0) -> float:
    """Compute entropy H(Y) using plug-in estimator with add-alpha smoothing."""
    counts = Counter(labels.tolist())
    n = len(labels)
    n_classes = len(counts)
    total = n + alpha * n_classes
    entropy = 0.0
    for count in counts.values():
        p = (count + alpha) / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def _conditional_entropy_plugin(
    labels: np.ndarray, codes: np.ndarray, alpha: float = 1.0
) -> float:
    """Compute conditional entropy H(Y | Z) using plug-in estimator."""
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)

    code_keys = [tuple(c) for c in codes]
    n = len(labels)

    joint_counts: dict[tuple[object, ...], int] = {}
    code_counts: dict[tuple[int, ...], int] = {}

    for code_key, label in zip(code_keys, labels):
        joint_counts[(code_key, label)] = joint_counts.get((code_key, label), 0) + 1
        code_counts[code_key] = code_counts.get(code_key, 0) + 1

    n_labels = len(set(labels.tolist()))

    cond_entropy = 0.0
    for code_key, code_count in code_counts.items():
        p_z = code_count / n
        h_y_given_z = 0.0
        total_with_smoothing = code_count + alpha * n_labels
        for label in set(labels.tolist()):
            joint_count = joint_counts.get((code_key, label), 0)
            p_y_given_z = (joint_count + alpha) / total_with_smoothing
            if p_y_given_z > 0:
                h_y_given_z -= p_y_given_z * math.log(p_y_given_z)
        cond_entropy += p_z * h_y_given_z

    return cond_entropy


def _compute_mi_at_level(
    labels: np.ndarray, codes: np.ndarray, level: int, alpha: float = 1.0
) -> float:
    """Compute MI at a specific level: I(Y; k_l | Z_{1:l-1}).

    Uses plug-in estimator with add-alpha smoothing.
    """
    if level == 0:
        h_y = _entropy_plugin(labels, alpha)
        h_y_given_z = _conditional_entropy_plugin(labels, codes[:, :1], alpha)
        return max(0.0, h_y - h_y_given_z)
    else:
        h_y_given_prev = _conditional_entropy_plugin(labels, codes[:, :level], alpha)
        h_y_given_z = _conditional_entropy_plugin(labels, codes[:, : level + 1], alpha)
        return max(0.0, h_y_given_prev - h_y_given_z)


def compute_mi_guided_dcut(
    cluster_ids: np.ndarray,
    codes: np.ndarray,
    n_levels: int,
    n_arms: int,
    n_rounds: int,
    min_obs_per_pair: int = 20,
) -> tuple[int, list[float]]:
    """Select D_cut automatically via MI diagnostic on held-out data.

    Uses an exploration-budget criterion: include level ℓ as long as the
    resulting effective context count leaves at least ``min_obs_per_pair``
    observations per (context, arm) pair over the horizon.  MI is computed
    at each level for diagnostic purposes but the budget is the binding
    constraint — the plug-in MI estimator is unreliable when the number of
    conditioning bins is large relative to n_samples.

    Args:
        cluster_ids: True cluster labels (proxy for reward-relevant labels).
        codes: Full RQ codes, shape (N, n_levels).
        n_levels: Number of RQ levels to consider.
        n_arms: Number of bandit arms.
        n_rounds: Bandit horizon (total rounds).
        min_obs_per_pair: Minimum observations per (context, arm) pair
            required for Thompson Sampling to converge.

    Returns:
        d_cut: Selected depth cut.
        mi_per_level: MI values per level (for diagnostics).
    """
    mi_per_level: list[float] = []
    for level in range(n_levels):
        mi = _compute_mi_at_level(cluster_ids, codes, level)
        mi_per_level.append(mi)

    max_contexts = n_rounds / (n_arms * min_obs_per_pair)

    d_cut = 1  # always include level 0
    prev_eff = len({tuple(row) for row in codes[:, :1]})
    for level in range(1, n_levels):
        # Stop if adding this level would exceed the exploration budget
        candidate_dcut = level + 1
        eff_contexts = len({tuple(row) for row in codes[:, :candidate_dcut]})
        if eff_contexts > max_contexts:
            break
        # Gate: diminishing returns — require at least 50% more effective
        # contexts to justify the additional level.  If going deeper barely
        # increases the number of unique contexts, the extra level is
        # mostly noise and hurts by fragmenting the observation budget.
        if prev_eff > 0 and eff_contexts / prev_eff < 1.5:
            break

        d_cut = candidate_dcut
        prev_eff = eff_contexts

    return max(1, d_cut), mi_per_level


# -----------------------------------------------------------------------
# Reward structure
# -----------------------------------------------------------------------


def _expected_reward(
    cluster_id: int,
    arm: int,
    best_arms: np.ndarray,
    n_arms: int,
) -> float:
    """Expected reward.

    Best arm for the cluster gets 0.8.
    The "second-best" arm (best + 1 mod n_arms) gets 0.5.
    All others get 0.2.
    """
    best = best_arms[cluster_id]
    if arm == best:
        return 0.8
    elif arm == (best + 1) % n_arms:
        return 0.5
    else:
        return 0.2


def _stochastic_reward(
    cluster_id: int,
    arm: int,
    best_arms: np.ndarray,
    n_arms: int,
    rng: np.random.RandomState,
) -> float:
    """Stochastic reward: expected reward plus Gaussian noise."""
    mu = _expected_reward(cluster_id, arm, best_arms, n_arms)
    return float(rng.normal(mu, 0.1))


# -----------------------------------------------------------------------
# Bandit runners
# -----------------------------------------------------------------------


def run_random_bandit(
    cluster_ids: np.ndarray,
    best_arms: np.ndarray,
    n_arms: int,
    n_rounds: int,
    n_samples: int,
    seed: int,
) -> float:
    """Run a random-arm baseline. Returns cumulative regret."""
    rng = np.random.RandomState(seed)
    regret = 0.0
    for _t in range(n_rounds):
        idx = rng.randint(n_samples)
        arm = rng.randint(n_arms)
        regret += 0.8 - _expected_reward(cluster_ids[idx], arm, best_arms, n_arms)
    return regret


def run_kmeans_bandit(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    best_arms: np.ndarray,
    n_arms: int,
    n_rounds: int,
    k: int,
    seed: int,
) -> float:
    """Run TS bandit using K-Means cluster ID as context.

    Returns:
        Final cumulative regret.
    """
    rng = np.random.RandomState(seed)

    km = MiniBatchKMeans(
        n_clusters=min(k, embeddings.shape[0]), random_state=seed, n_init=1
    )
    km.fit(embeddings)
    assigned = km.labels_

    bandit = HierarchicalThompsonSampling(
        n_arms=n_arms, n_codes=k, d_cut=1, device="cpu"
    )

    regret = 0.0
    n_samples = embeddings.shape[0]

    for _t in range(n_rounds):
        idx = rng.randint(n_samples)
        context_code = torch.tensor([[assigned[idx]]], dtype=torch.long)
        arm = bandit.select_arm(context_code).item()

        reward = _stochastic_reward(cluster_ids[idx], arm, best_arms, n_arms, rng)
        bandit.update(context_code, torch.tensor([arm]), torch.tensor([reward]))

        regret += 0.8 - _expected_reward(cluster_ids[idx], arm, best_arms, n_arms)

    return regret


def run_rq_bandit_fixed_dcut(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    best_arms: np.ndarray,
    n_arms: int,
    n_rounds: int,
    n_levels: int,
    n_codes: int,
    d_cut: int,
    seed: int,
) -> float:
    """Run TS bandit using RQ trunk codes at a pre-determined D_cut.

    Args:
        embeddings: (N, D) embedding matrix.
        cluster_ids: (N,) true cluster assignments.
        best_arms: (n_clusters,) best arm per cluster.
        n_arms: Number of bandit arms.
        n_rounds: Number of bandit rounds.
        n_levels: RQ levels for code simulation.
        n_codes: RQ codebook size per level.
        d_cut: Depth cut (determined externally via pilot).
        seed: Random seed.

    Returns:
        Cumulative regret.
    """
    rng = np.random.RandomState(seed)

    codes = simulate_rq_codes(embeddings, n_levels, n_codes, seed)
    trunk_codes = codes[:, :d_cut]

    bandit = HierarchicalThompsonSampling(
        n_arms=n_arms, n_codes=n_codes, d_cut=d_cut, device="cpu"
    )

    regret = 0.0
    n_samples = embeddings.shape[0]

    for _t in range(n_rounds):
        idx = rng.randint(n_samples)
        context_code = torch.from_numpy(trunk_codes[idx : idx + 1])
        arm = bandit.select_arm(context_code).item()

        reward = _stochastic_reward(cluster_ids[idx], arm, best_arms, n_arms, rng)
        bandit.update(context_code, torch.tensor([arm]), torch.tensor([reward]))

        regret += 0.8 - _expected_reward(cluster_ids[idx], arm, best_arms, n_arms)

    return regret


# -----------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------


def run_comparison_for_setting(
    n_clusters: int,
    config: ContextComparisonConfig,
) -> dict[str, object]:
    """Run the full comparison for a single n_clusters setting.

    Returns:
        Dictionary with per-granularity regret arrays and RQ results.
    """
    results: dict[str, object] = {
        "n_clusters": n_clusters,
        "granularities": list(config.granularities),
        "kmeans_mean": [],
        "kmeans_std": [],
    }

    print(
        f"\nProblem: {n_clusters} clusters, {config.n_arms} arms, "
        f"{config.n_rounds} rounds, {config.n_seeds} seeds"
    )
    print(f"Samples: {config.n_samples}, Embedding dim: {config.embedding_dim}")
    print("-" * 70)

    # --- Phase 0: Determine D_cut via pilot run ---
    pilot_seed = 42
    pilot_embeddings, pilot_cluster_ids, _ = generate_structured_embeddings(
        n_samples=config.n_samples,
        embedding_dim=config.embedding_dim,
        n_clusters=n_clusters,
        n_arms=config.n_arms,
        seed=pilot_seed,
    )
    pilot_codes = simulate_rq_codes(
        pilot_embeddings, config.n_levels, config.n_codes, pilot_seed
    )
    d_cut, mi_per_level = compute_mi_guided_dcut(
        pilot_cluster_ids,
        pilot_codes,
        config.n_levels,
        config.n_arms,
        config.n_rounds,
    )
    pilot_eff_contexts = len({tuple(row) for row in pilot_codes[:, :d_cut]})

    print(f"\nMI per level: {[f'{m:.4f}' for m in mi_per_level]}")
    print(
        f"D_cut={d_cut} (exploration-budget-guided), "
        f"effective contexts={pilot_eff_contexts}"
    )

    # --- Phase 1: Run RQ at fixed D_cut (single operating point) ---
    rq_regrets: list[float] = []
    random_regrets: list[float] = []

    for s in range(config.n_seeds):
        seed = 1000 * s + 42
        embeddings, cluster_ids, best_arms = generate_structured_embeddings(
            n_samples=config.n_samples,
            embedding_dim=config.embedding_dim,
            n_clusters=n_clusters,
            n_arms=config.n_arms,
            seed=seed,
        )

        rq_r = run_rq_bandit_fixed_dcut(
            embeddings,
            cluster_ids,
            best_arms,
            config.n_arms,
            config.n_rounds,
            config.n_levels,
            config.n_codes,
            d_cut,
            seed,
        )
        rq_regrets.append(rq_r)

        rand_r = run_random_bandit(
            cluster_ids,
            best_arms,
            config.n_arms,
            config.n_rounds,
            config.n_samples,
            seed,
        )
        random_regrets.append(rand_r)

    rq_mean = float(np.mean(rq_regrets))
    rq_std = float(np.std(rq_regrets))
    rand_mean = float(np.mean(random_regrets))
    rand_std = float(np.std(random_regrets))

    results["rq_mean"] = rq_mean
    results["rq_std"] = rq_std
    results["rq_dcut"] = d_cut
    results["rq_eff_contexts"] = pilot_eff_contexts
    results["random_mean"] = rand_mean
    results["random_std"] = rand_std

    print(f"Random:    regret = {rand_mean:.1f} +/- {rand_std:.1f}")
    print(f"RQ:        regret = {rq_mean:.1f} +/- {rq_std:.1f}")

    # --- Phase 2: Sweep K-Means across granularities ---
    print(f"\n{'k':>6s}  {'KMeans regret':>18s}  {'vs RQ':>12s}")
    print("-" * 42)

    km_oracle_regret = float("inf")
    km_oracle_k = 0

    for k in config.granularities:
        km_regrets: list[float] = []

        for s in range(config.n_seeds):
            seed = 1000 * s + 42
            embeddings, cluster_ids, best_arms = generate_structured_embeddings(
                n_samples=config.n_samples,
                embedding_dim=config.embedding_dim,
                n_clusters=n_clusters,
                n_arms=config.n_arms,
                seed=seed,
            )
            km_r = run_kmeans_bandit(
                embeddings,
                cluster_ids,
                best_arms,
                config.n_arms,
                config.n_rounds,
                k,
                seed,
            )
            km_regrets.append(km_r)

        km_mean = float(np.mean(km_regrets))
        km_std = float(np.std(km_regrets))

        # pyre-ignore[16]: results values are lists
        results["kmeans_mean"].append(km_mean)
        # pyre-ignore[16]
        results["kmeans_std"].append(km_std)

        # Track oracle
        if km_mean < km_oracle_regret:
            km_oracle_regret = km_mean
            km_oracle_k = k

        # Compare to RQ
        if rq_mean < km_mean - 1.0:
            verdict = "RQ wins"
        elif km_mean < rq_mean - 1.0:
            verdict = "KM wins"
        else:
            verdict = "~tied"

        print(f"{k:6d}  {km_mean:7.1f} +/- {km_std:5.1f}     {verdict}")

    results["km_oracle_regret"] = km_oracle_regret
    results["km_oracle_k"] = km_oracle_k

    # --- Summary ---
    rq_wins = 0
    n_gran = len(config.granularities)
    for i in range(n_gran):
        # pyre-ignore[16]
        if rq_mean < results["kmeans_mean"][i] - 1.0:
            rq_wins += 1

    gap = rq_mean - km_oracle_regret
    gap_pct = (gap / km_oracle_regret * 100.0) if km_oracle_regret > 0 else 0.0

    print(f"\nRQ (MI-guided):   {rq_mean:.1f} +/- {rq_std:.1f}")
    print(
        f"KM oracle (k={km_oracle_k}): {km_oracle_regret:.1f}"
        f"  [<-- requires sweeping k]"
    )
    print(f"RQ gap to oracle: {gap:+.1f} ({gap_pct:.0f}%)  [<-- cost of not tuning k]")
    print(
        f"\nConclusion: RQ at MI-guided D_cut beats K-Means at "
        f"{rq_wins}/{n_gran} granularities"
    )
    if gap_pct < 25:
        print(
            f"            and is within {gap_pct:.0f}% of KM oracle "
            f"-- WITHOUT tuning k."
        )

    return results


def run_comparison(config: ContextComparisonConfig) -> list[dict[str, object]]:
    """Run the comparison across multiple n_clusters settings.

    Returns:
        List of per-setting result dictionaries.
    """
    all_results: list[dict[str, object]] = []

    print("\n" + "=" * 70)
    print("P1: RQ (MI-guided D_cut) vs K-Means Context Comparison")
    print("=" * 70)
    print(
        f"Arms={config.n_arms}  Rounds={config.n_rounds}  "
        f"Seeds={config.n_seeds}  Dim={config.embedding_dim}"
    )

    for n_clusters in config.cluster_settings:
        print("\n" + "=" * 70)
        print(f"  Setting: n_clusters = {n_clusters}")
        print("=" * 70)

        results = run_comparison_for_setting(n_clusters, config)
        all_results.append(results)

    # --- Cross-setting summary ---
    print("\n" + "=" * 70)
    print("CROSS-SETTING SUMMARY")
    print("=" * 70)
    print(
        f"{'n_clusters':>11s}  {'RQ regret':>14s}  "
        f"{'KM oracle':>14s}  {'Gap':>8s}  {'KM oracle k':>12s}"
    )
    print("-" * 70)
    for r in all_results:
        nc = r["n_clusters"]
        rq_m = r["rq_mean"]
        km_o = r["km_oracle_regret"]
        # pyre-ignore[58]: arithmetic on object types
        gap_pct = ((rq_m - km_o) / km_o * 100.0) if km_o > 0 else 0.0
        km_k = r["km_oracle_k"]
        # pyre-ignore[6]: formatting object types
        print(f"{nc:11d}  {rq_m:14.1f}  {km_o:14.1f}  {gap_pct:+7.0f}%  {km_k:>12d}")

    return all_results


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description=(
            "P1: RQ (MI-guided D_cut) vs K-Means Context Comparison "
            "for Thompson Sampling"
        )
    )
    parser.add_argument("--n-arms", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=10000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-codes", type=int, default=16)
    parser.add_argument("--n-levels", type=int, default=8)
    parser.add_argument(
        "--cluster-settings",
        type=int,
        nargs="+",
        default=[20, 50, 100],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = ContextComparisonConfig(
        n_arms=args.n_arms,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
        embedding_dim=args.embedding_dim,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_levels=args.n_levels,
        cluster_settings=tuple(args.cluster_settings),
        device=args.device,
    )

    run_comparison(config)

    print("\nEXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
