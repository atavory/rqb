#!/usr/bin/env python3

"""Neural Thompson Sampling experiments using RQ trunk addresses.

This script runs experiments comparing:
1. HierarchicalThompsonSampling (O(1) bandit using RQ trunk addresses)
2. NeuralTSBaseline (neural network baseline with MC Dropout)

Experiments:
- Latency benchmark: decision time comparison
- Regret comparison: cumulative regret over rounds (fixed context pool)
- Exploration strategy: Thompson Sampling vs UCB vs Epsilon-Greedy
"""

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from modules.bandits import (
    BanditConfig,
    ExplorationStrategy,
    HierarchicalThompsonSampling,
    NeuralTSBaseline,
)


@dataclass
class ExperimentConfig:
    """Configuration for NTS experiments."""

    n_arms: int = 10
    n_rounds: int = 10000
    embedding_dim: int = 128
    n_codes: int = 256  # Codebook size per RQ level
    d_cut: int = 2  # Number of trunk levels
    n_contexts: int = 50  # Number of unique contexts in the pool
    batch_sizes: tuple = (1, 10, 100, 1000)
    n_mc_samples: int = 20
    device: str = "cpu"
    seed: int = 42
    n_seeds: int = 10


def generate_synthetic_codes(
    n_samples: int, n_codes: int, d_cut: int, device: str
) -> torch.Tensor:
    """Generate synthetic RQ codes for testing (used by latency benchmark)."""
    codes = torch.randint(0, n_codes, (n_samples, d_cut), device=device)
    return codes


def generate_context_pool(
    n_contexts: int,
    n_codes: int,
    d_cut: int,
    n_arms: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate a fixed pool of contexts with per-context arm rewards.

    Creates a set of unique context codes and assigns each context a
    different arm reward distribution via Beta(2, 5) draws.  Different
    contexts will have different best arms, making contextual information
    valuable for the bandit.

    Args:
        n_contexts: Desired number of unique contexts.
        n_codes: Codebook size per RQ level.
        d_cut: Number of trunk levels forming the context key.
        n_arms: Number of bandit arms.
        seed: Random seed for reproducibility.

    Returns:
        codes: (n_unique, d_cut) int tensor of unique context codes.
        true_rewards: (n_unique, n_arms) reward probabilities per context.
    """
    rng = np.random.RandomState(seed)
    # Use a small effective codebook so we can generate enough unique
    # contexts without needing an enormous codebook.
    eff_k = min(n_codes, max(4, int(math.ceil(n_contexts ** (1.0 / d_cut)))))
    codes = torch.from_numpy(rng.randint(0, eff_k, (n_contexts * 2, d_cut)))
    # Deduplicate to get unique contexts
    unique_codes = torch.unique(codes, dim=0)
    # Trim to at most n_contexts
    if unique_codes.shape[0] > n_contexts:
        unique_codes = unique_codes[:n_contexts]
    # Per-context reward probabilities
    true_rewards = rng.beta(2, 5, size=(unique_codes.shape[0], n_arms))
    return unique_codes, true_rewards


def run_latency_benchmark(config: ExperimentConfig) -> dict:
    """Benchmark decision latency: HierarchicalTS vs NeuralTS."""
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)

    # Create HierarchicalTS bandit
    hierarchical_ts = HierarchicalThompsonSampling(
        n_arms=config.n_arms,
        n_codes=config.n_codes,
        d_cut=config.d_cut,
    )

    # Create NeuralTS baseline
    neural_ts = NeuralTSBaseline(
        input_dim=config.embedding_dim,
        hidden_dim=128,
        n_arms=config.n_arms,
        n_mc_samples=config.n_mc_samples,
    )

    results: dict = {"batch_size": [], "hierarchical_ms": [], "neural_ms": []}

    for batch_size in config.batch_sizes:
        # Generate synthetic codes for HierarchicalTS
        codes = generate_synthetic_codes(
            batch_size, config.n_codes, config.d_cut, config.device
        )
        # Generate synthetic embeddings for NeuralTS
        embeddings = torch.randn(batch_size, config.embedding_dim, device=config.device)

        # Benchmark HierarchicalTS
        start = time.perf_counter()
        for _ in range(100):
            hierarchical_ts.select_arm(codes)
        hierarchical_time = (time.perf_counter() - start) / 100 * 1000

        # Benchmark NeuralTS
        start = time.perf_counter()
        for _ in range(100):
            neural_ts.select_arm(embeddings)
        neural_time = (time.perf_counter() - start) / 100 * 1000

        results["batch_size"].append(batch_size)
        results["hierarchical_ms"].append(hierarchical_time)
        results["neural_ms"].append(neural_time)

        speedup = neural_time / hierarchical_time
        print(
            f"Batch {batch_size:5d}: Hierarchical={hierarchical_time:.3f}ms, "
            f"Neural={neural_time:.3f}ms, Speedup={speedup:.1f}x"
        )

    return results


def run_regret_experiment(config: ExperimentConfig) -> dict:
    """Compare cumulative regret: HierarchicalTS vs random.

    Uses a fixed pool of contexts with per-context reward distributions
    so that TS can learn context-specific arm quality over repeated
    observations.
    """
    print("\n" + "=" * 60)
    print("REGRET EXPERIMENT")
    print("=" * 60)

    all_regret_ts: list[list[float]] = []
    all_regret_random: list[list[float]] = []

    for seed_idx in range(config.n_seeds):
        seed = config.seed + seed_idx * 1000

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Fixed context pool with per-context rewards
        context_codes, true_rewards = generate_context_pool(
            n_contexts=config.n_contexts,
            n_codes=config.n_codes,
            d_cut=config.d_cut,
            n_arms=config.n_arms,
            seed=seed,
        )
        n_ctx = context_codes.shape[0]
        best_rewards = true_rewards.max(axis=1)  # (n_ctx,)

        rng = np.random.RandomState(seed + 1)

        # Create bandit
        hierarchical_ts = HierarchicalThompsonSampling(
            n_arms=config.n_arms,
            n_codes=config.n_codes,
            d_cut=config.d_cut,
        )

        seed_regret_ts: list[float] = []
        seed_regret_random: list[float] = []
        regret_ts = 0.0
        regret_random = 0.0

        for round_idx in range(config.n_rounds):
            # Sample a context from the fixed pool
            ctx_idx = rng.randint(n_ctx)
            codes = context_codes[ctx_idx : ctx_idx + 1]

            # Thompson Sampling choice
            arm_ts = hierarchical_ts.select_arm(codes).item()
            reward_ts = float(rng.binomial(1, true_rewards[ctx_idx, arm_ts]))
            hierarchical_ts.update(
                codes, torch.tensor([arm_ts]), torch.tensor([reward_ts])
            )
            regret_ts += best_rewards[ctx_idx] - true_rewards[ctx_idx, arm_ts]

            # Random baseline
            arm_random = rng.randint(config.n_arms)
            regret_random += best_rewards[ctx_idx] - true_rewards[ctx_idx, arm_random]

            if (round_idx + 1) % 1000 == 0:
                seed_regret_ts.append(regret_ts)
                seed_regret_random.append(regret_random)

        all_regret_ts.append(seed_regret_ts)
        all_regret_random.append(seed_regret_random)

    # Compute mean/std across seeds
    ts_array = np.array(all_regret_ts)  # (n_seeds, n_checkpoints)
    rand_array = np.array(all_regret_random)
    ts_mean = ts_array.mean(axis=0)
    ts_std = ts_array.std(axis=0)
    rand_mean = rand_array.mean(axis=0)
    rand_std = rand_array.std(axis=0)

    rounds = list(range(1000, config.n_rounds + 1, 1000))
    print(f"\nResults across {config.n_seeds} seeds:")
    print(
        f"{'Round':>7s}  {'TS mean':>10s}  {'TS std':>8s}  "
        f"{'Rand mean':>10s}  {'Rand std':>9s}  {'Ratio':>6s}"
    )
    print("-" * 60)
    for i, r in enumerate(rounds):
        ratio = rand_mean[i] / ts_mean[i] if ts_mean[i] > 0 else float("inf")
        print(
            f"{r:7d}  {ts_mean[i]:10.1f}  {ts_std[i]:8.1f}  "
            f"{rand_mean[i]:10.1f}  {rand_std[i]:9.1f}  {ratio:6.2f}x"
        )

    return {
        "rounds": rounds,
        "regret_ts_mean": ts_mean.tolist(),
        "regret_ts_std": ts_std.tolist(),
        "regret_random_mean": rand_mean.tolist(),
        "regret_random_std": rand_std.tolist(),
        "n_seeds": config.n_seeds,
    }


def run_exploration_comparison(config: ExperimentConfig) -> dict:
    """Compare exploration strategies with context-dependent rewards.

    Uses a fixed context pool so that all strategies face the same
    contextual bandit problem.
    """
    print("\n" + "=" * 60)
    print("EXPLORATION STRATEGY COMPARISON")
    print("=" * 60)

    strategies = [
        ExplorationStrategy.THOMPSON_SAMPLING,
        ExplorationStrategy.UCB,
        ExplorationStrategy.EPSILON_GREEDY,
    ]

    results: dict = {
        str(s): {"mean": 0.0, "std": 0.0, "per_seed": []} for s in strategies
    }

    for strategy in strategies:
        seed_regrets: list[float] = []

        for seed_idx in range(config.n_seeds):
            seed = config.seed + seed_idx * 1000

            np.random.seed(seed)
            torch.manual_seed(seed)

            # Same fixed context pool for all strategies
            context_codes, true_rewards = generate_context_pool(
                n_contexts=config.n_contexts,
                n_codes=config.n_codes,
                d_cut=config.d_cut,
                n_arms=config.n_arms,
                seed=seed,
            )
            n_ctx = context_codes.shape[0]
            best_rewards = true_rewards.max(axis=1)

            rng = np.random.RandomState(seed + 1)

            bandit_config = BanditConfig(
                exploration=strategy,
                epsilon=0.1,
                ucb_confidence=2.0,
            )

            bandit = HierarchicalThompsonSampling(
                n_arms=config.n_arms,
                n_codes=config.n_codes,
                d_cut=config.d_cut,
                config=bandit_config,
            )
            regret = 0.0

            for _round_idx in range(config.n_rounds):
                ctx_idx = rng.randint(n_ctx)
                codes = context_codes[ctx_idx : ctx_idx + 1]
                arm = bandit.select_arm(codes).item()
                reward = float(rng.binomial(1, true_rewards[ctx_idx, arm]))
                bandit.update(codes, torch.tensor([arm]), torch.tensor([reward]))
                regret += best_rewards[ctx_idx] - true_rewards[ctx_idx, arm]

            seed_regrets.append(regret)

        mean_regret = float(np.mean(seed_regrets))
        std_regret = float(np.std(seed_regrets))
        results[str(strategy)]["mean"] = mean_regret
        results[str(strategy)]["std"] = std_regret
        results[str(strategy)]["per_seed"] = seed_regrets

        print(
            f"{strategy}: regret = {mean_regret:.1f} +/- {std_regret:.1f} "
            f"({config.n_seeds} seeds)"
        )

    return results


def main() -> None:
    """Run all NTS experiments."""
    parser = argparse.ArgumentParser(description="Neural Thompson Sampling Experiments")
    parser.add_argument(
        "--experiment",
        choices=["latency", "regret", "exploration", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on",
    )
    parser.add_argument("--n-rounds", type=int, default=10000, help="Number of rounds")
    parser.add_argument("--n-arms", type=int, default=100, help="Number of arms")
    parser.add_argument("--n-contexts", type=int, default=50, help="Context pool size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_arms=args.n_arms,
        n_rounds=args.n_rounds,
        n_contexts=args.n_contexts,
        device=args.device,
        seed=args.seed,
        n_seeds=args.n_seeds,
    )

    print("=" * 60)
    print("NEURAL THOMPSON SAMPLING EXPERIMENTS")
    print("=" * 60)
    print(f"Config: {config}")

    if args.experiment in ["latency", "all"]:
        run_latency_benchmark(config)

    if args.experiment in ["regret", "all"]:
        run_regret_experiment(config)

    if args.experiment in ["exploration", "all"]:
        run_exploration_comparison(config)

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
