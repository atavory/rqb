

"""Cold-start transfer experiment for hierarchical Thompson Sampling.

Validates that ColdStartTransfer reduces regret on unseen contexts by
bootstrapping NIG priors from previously-seen (similar) contexts.

Design:
    1. Warm-up phase: train a bandit on K_seen contexts for T_warmup rounds.
    2. Cold-start phase: introduce K_new unseen contexts.
    3. Compare three strategies over T_cold rounds on the new contexts:
       - No transfer: new contexts start with uninformative priors.
       - Full copy (shrinkage=0): copy priors from nearest seen context.
       - Shrunk (shrinkage=0.5): blend nearest context prior with uninformative.
    4. Report cumulative regret on new contexts across seeds.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from modules.bandits import (
    ColdStartTransfer,
    HierarchicalThompsonSampling,
)


@dataclass
class ColdStartConfig:
    """Configuration for the cold-start transfer experiment."""

    n_arms: int = 10
    n_codes: int = 16
    d_cut: int = 2
    n_seen_contexts: int = 30
    n_new_contexts: int = 20
    t_warmup: int = 5000
    t_cold: int = 3000
    n_seeds: int = 10
    device: str = "cpu"


def _generate_context_pool(
    n_contexts: int,
    n_codes: int,
    d_cut: int,
    n_arms: int,
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate a pool of unique context codes with per-context rewards."""
    eff_k = min(n_codes, max(4, int(math.ceil(n_contexts ** (1.0 / d_cut)))))
    codes = torch.from_numpy(rng.randint(0, eff_k, (n_contexts * 3, d_cut)))
    unique_codes = torch.unique(codes, dim=0)
    if unique_codes.shape[0] > n_contexts:
        unique_codes = unique_codes[:n_contexts]
    true_rewards = rng.beta(2, 5, size=(unique_codes.shape[0], n_arms))
    return unique_codes, true_rewards


def _find_nearest_seen_context(
    new_code: torch.Tensor,
    seen_codes: torch.Tensor,
) -> int:
    """Find the index of the nearest seen context (L1 distance)."""
    dists = (seen_codes - new_code.unsqueeze(0)).abs().sum(dim=1)
    return int(dists.argmin().item())


def run_cold_start_experiment(config: ColdStartConfig) -> dict[str, object]:
    """Run the cold-start transfer experiment.

    Returns a results dict with per-strategy mean/std regret on new contexts.
    """
    print("=" * 60)
    print("COLD-START TRANSFER EXPERIMENT")
    print("=" * 60)
    print(f"Config: {config}")

    strategies = {
        "no_transfer": None,
        "full_copy": 0.0,
        "shrunk_0.5": 0.5,
    }

    all_results: dict[str, list[float]] = {name: [] for name in strategies}

    for seed_idx in range(config.n_seeds):
        seed = 42 + seed_idx * 1000
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        # Generate seen + new context pools (non-overlapping)
        total_contexts = config.n_seen_contexts + config.n_new_contexts
        all_codes, all_rewards = _generate_context_pool(
            n_contexts=total_contexts,
            n_codes=config.n_codes,
            d_cut=config.d_cut,
            n_arms=config.n_arms,
            rng=rng,
        )
        n_total = all_codes.shape[0]
        n_seen = min(config.n_seen_contexts, n_total // 2)
        seen_codes = all_codes[:n_seen]
        seen_rewards = all_rewards[:n_seen]
        new_codes = all_codes[n_seen:]
        new_rewards = all_rewards[n_seen:]
        n_new = new_codes.shape[0]
        best_new_rewards = new_rewards.max(axis=1)

        # --- Warm-up: train bandit on seen contexts ---
        warm_bandit = HierarchicalThompsonSampling(
            n_arms=config.n_arms,
            n_codes=config.n_codes,
            d_cut=config.d_cut,
            device=config.device,
        )
        for _t in range(config.t_warmup):
            idx = rng.randint(n_seen)
            codes = seen_codes[idx : idx + 1]
            arm = warm_bandit.select_arm(codes).item()
            reward = float(rng.binomial(1, seen_rewards[idx, arm]))
            warm_bandit.update(codes, torch.tensor([arm]), torch.tensor([reward]))

        # --- Cold-start phase: test each transfer strategy ---
        for name, shrinkage in strategies.items():
            cold_rng = np.random.RandomState(seed + 1)
            torch.manual_seed(seed + 1)

            cold_bandit = HierarchicalThompsonSampling(
                n_arms=config.n_arms,
                n_codes=config.n_codes,
                d_cut=config.d_cut,
                device=config.device,
            )

            if shrinkage is not None:
                # Transfer priors from nearest seen context
                transfer = ColdStartTransfer(
                    n_arms=config.n_arms,
                    shrinkage=shrinkage,
                    device=config.device,
                )
                for i in range(n_new):
                    nearest_idx = _find_nearest_seen_context(new_codes[i], seen_codes)
                    key = tuple(seen_codes[nearest_idx].tolist())
                    if key in warm_bandit._contexts:
                        source_stats = warm_bandit._contexts[key]
                        new_key = tuple(new_codes[i].tolist())
                        cold_bandit._contexts[new_key] = transfer.transfer(source_stats)

            # Run cold-start rounds
            regret = 0.0
            for _t in range(config.t_cold):
                idx = cold_rng.randint(n_new)
                codes = new_codes[idx : idx + 1]
                arm = cold_bandit.select_arm(codes).item()
                reward = float(cold_rng.binomial(1, new_rewards[idx, arm]))
                cold_bandit.update(codes, torch.tensor([arm]), torch.tensor([reward]))
                regret += best_new_rewards[idx] - new_rewards[idx, arm]

            all_results[name].append(regret)

    # --- Report ---
    print(f"\nResults across {config.n_seeds} seeds:")
    print(f"{'Strategy':>20s}  {'Mean regret':>12s}  {'Std':>8s}")
    print("-" * 46)
    for name in strategies:
        arr = np.array(all_results[name])
        print(f"{name:>20s}  {arr.mean():12.1f}  {arr.std():8.1f}")

    # Improvement summary
    no_transfer_mean = np.mean(all_results["no_transfer"])
    for name in ["full_copy", "shrunk_0.5"]:
        mean = np.mean(all_results[name])
        pct = (no_transfer_mean - mean) / no_transfer_mean * 100.0
        print(f"\n{name} reduces cold-start regret by {pct:.1f}% vs no_transfer")

    return {
        "strategies": list(strategies.keys()),
        "results": {
            name: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "per_seed": v,
            }
            for name, v in all_results.items()
        },
    }


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Cold-Start Transfer Experiment for Hierarchical TS"
    )
    parser.add_argument("--n-arms", type=int, default=10)
    parser.add_argument("--n-codes", type=int, default=16)
    parser.add_argument("--d-cut", type=int, default=2)
    parser.add_argument("--n-seen-contexts", type=int, default=30)
    parser.add_argument("--n-new-contexts", type=int, default=20)
    parser.add_argument("--t-warmup", type=int, default=5000)
    parser.add_argument("--t-cold", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = ColdStartConfig(
        n_arms=args.n_arms,
        n_codes=args.n_codes,
        d_cut=args.d_cut,
        n_seen_contexts=args.n_seen_contexts,
        n_new_contexts=args.n_new_contexts,
        t_warmup=args.t_warmup,
        t_cold=args.t_cold,
        n_seeds=args.n_seeds,
        device=args.device,
    )

    run_cold_start_experiment(config)

    print("\nEXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
