#!/usr/bin/env python3
"""Synthetic crossover experiment: find the horizon T where d=2 beats d=1.

Generates synthetic contexts, trains RQ codebooks, runs contextual bandits
at d=1, d=2, d=3 to T=1M+ and records regret curves. Finds the crossover
point where deeper RQ overtakes shallow.

Usage:
    python3 scripts/synthetic_crossover.py
    python3 scripts/synthetic_crossover.py --T 10000000 --seeds 20 --b 4
"""

import argparse
import csv
import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


def make_reward_function(D: int, n_arms: int, seed: int = 0):
    """Create an additive nonlinear reward function with local structure.

    r(x, a) = sum_i bump_i(x_i, a) + noise

    Each bump_i is a piecewise function that depends on which "bin" x_i falls
    into, creating local reward structure that benefits from fine partitioning.
    """
    rng = np.random.RandomState(seed)

    # For each dimension, create per-arm reward offsets at 4 breakpoints
    n_bins = 8
    breakpoints = np.linspace(-3, 3, n_bins + 1)
    # reward_table[dim, bin, arm] = reward contribution
    reward_table = rng.randn(D, n_bins, n_arms) * 0.3

    def reward(x, arm):
        """Compute reward for context x and arm choice."""
        r = 0.0
        for d in range(D):
            bin_idx = np.searchsorted(breakpoints[1:], x[d])
            bin_idx = min(bin_idx, n_bins - 1)
            r += reward_table[d, bin_idx, arm]
        return r

    def optimal_reward(x):
        """Compute the best possible reward (oracle)."""
        rewards = [reward(x, a) for a in range(n_arms)]
        return max(rewards)

    def optimal_arm(x):
        """Return the optimal arm for context x."""
        rewards = [reward(x, a) for a in range(n_arms)]
        return int(np.argmax(rewards))

    return reward, optimal_reward, optimal_arm


def train_rq_codebook(X: np.ndarray, b: int, max_depth: int, seed: int = 42):
    """Train RQ codebook: iterative k-means on residuals.

    Returns list of (centroids, assignments) per level.
    """
    levels = []
    residual = X.copy()
    for d in range(max_depth):
        km = KMeans(n_clusters=b, n_init=3, random_state=seed + d)
        assignments = km.fit_predict(residual)
        centroids = km.cluster_centers_
        residual = residual - centroids[assignments]
        levels.append(centroids)
    return levels


def encode_rq(x: np.ndarray, levels: list[np.ndarray], depth: int) -> np.ndarray:
    """Encode a single context x at given depth. Returns l-hot indicator vector."""
    b = levels[0].shape[0]
    indicators = np.zeros(depth * b)
    residual = x.copy()
    for d in range(depth):
        dists = np.sum((levels[d] - residual) ** 2, axis=1)
        assignment = np.argmin(dists)
        indicators[d * b + assignment] = 1.0
        residual = residual - levels[d][assignment]
    return indicators


class LinTSAgent:
    """Linear Thompson Sampling on encoded features."""

    def __init__(self, d_feat: int, n_arms: int, nu: float = 1.0, lam: float = 1.0):
        self.n_arms = n_arms
        self.d = d_feat
        self.nu = nu
        self.lam = lam
        # Per-arm sufficient statistics
        self.B = [lam * np.eye(d_feat) for _ in range(n_arms)]
        self.f = [np.zeros(d_feat) for _ in range(n_arms)]
        self.theta_hat = [np.zeros(d_feat) for _ in range(n_arms)]

    def select_arm(self, phi: np.ndarray, rng: np.random.RandomState) -> int:
        """Thompson Sampling: sample from posterior using Cholesky."""
        best_arm = 0
        best_val = -np.inf
        for a in range(self.n_arms):
            # Sample: theta ~ N(theta_hat, nu^2 * B^{-1})
            # Use Cholesky of B, then solve for the sample
            L = np.linalg.cholesky(self.B[a])
            z = rng.randn(self.d)
            # B^{-1} = (L L^T)^{-1}, so B^{-1/2} z = L^{-T} z
            v = np.linalg.solve(L.T, z)
            theta_sample = self.theta_hat[a] + self.nu * v
            val = phi @ theta_sample
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, phi: np.ndarray, arm: int, reward: float):
        """Rank-1 update to sufficient statistics."""
        self.B[arm] += np.outer(phi, phi)
        self.f[arm] += reward * phi
        self.theta_hat[arm] = np.linalg.solve(self.B[arm], self.f[arm])


def run_bandit(
    reward_fn,
    optimal_reward_fn,
    levels: list[np.ndarray],
    depth: int,
    n_arms: int,
    D: int,
    T: int,
    seed: int,
    checkpoint_interval: int,
    nu: float = 1.0,
    lam: float = 1.0,
) -> list[tuple[int, float]]:
    """Run one bandit experiment. Returns [(checkpoint_T, cumulative_regret), ...]."""
    b = levels[0].shape[0]
    d_feat = depth * b
    agent = LinTSAgent(d_feat, n_arms, nu=nu, lam=lam)
    rng = np.random.RandomState(seed)

    cumulative_regret = 0.0
    checkpoints = []

    for t in range(1, T + 1):
        # Generate context
        x = rng.randn(D)

        # Encode
        phi = encode_rq(x, levels, depth)

        # Select arm
        arm = agent.select_arm(phi, rng)

        # Get reward
        r = reward_fn(x, arm) + rng.randn() * 0.1  # noise
        r_opt = optimal_reward_fn(x)
        cumulative_regret += (r_opt - r)

        # Update
        agent.update(phi, arm, r)

        # Checkpoint
        if t % checkpoint_interval == 0:
            checkpoints.append((t, cumulative_regret))

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Synthetic crossover experiment")
    parser.add_argument("--T", type=int, default=1_000_000, help="Horizon")
    parser.add_argument("--seeds", type=int, default=10, help="Number of bandit seeds")
    parser.add_argument("--b", type=int, default=4, help="Branching factor")
    parser.add_argument("--D", type=int, default=10, help="Context dimensionality")
    parser.add_argument("--n-arms", type=int, default=5, help="Number of arms")
    parser.add_argument("--max-depth", type=int, default=3, help="Max RQ depth")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000, help="Checkpoint every N steps")
    parser.add_argument("--nu", type=float, default=0.5, help="Thompson Sampling exploration parameter")
    parser.add_argument("--lam", type=float, default=1.0, help="Regularization")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    print(f"Config: T={args.T:,}, seeds={args.seeds}, b={args.b}, D={args.D}, "
          f"arms={args.n_arms}, max_depth={args.max_depth}, nu={args.nu}, lam={args.lam}")

    # Create reward function
    print("Creating reward function...", end="", flush=True)
    reward_fn, optimal_fn, _ = make_reward_function(args.D, args.n_arms, seed=999)
    print(" done")

    # Generate training contexts for codebook
    print("Training RQ codebook...", end="", flush=True)
    rng = np.random.RandomState(42)
    X_train = rng.randn(50_000, args.D)
    levels = train_rq_codebook(X_train, args.b, args.max_depth, seed=42)
    print(f" done ({args.max_depth} levels, b={args.b})")

    # Run experiments
    all_results = []

    for depth in range(1, args.max_depth + 1):
        d_feat = depth * args.b
        print(f"\n=== Depth {depth} (d_feat={d_feat}, cells={args.b**depth}) ===")

        for s in range(args.seeds):
            seed = s * 1000 + depth
            t0 = time.time()
            checkpoints = run_bandit(
                reward_fn, optimal_fn, levels, depth,
                args.n_arms, args.D, args.T, seed,
                args.checkpoint_interval, args.nu, args.lam,
            )
            elapsed = time.time() - t0
            final_regret = checkpoints[-1][1]
            print(f"  seed {s}: regret={final_regret:,.0f}, time={elapsed:.1f}s")

            for cp_t, cp_regret in checkpoints:
                all_results.append({
                    "depth": depth,
                    "seed": s,
                    "T": cp_t,
                    "cumulative_regret": cp_regret,
                })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Mean cumulative regret at final T")
    print(f"{'depth':<8} {'d_feat':<8} {'cells':<8} {'mean_regret':<15} {'std':<15}")
    for depth in range(1, args.max_depth + 1):
        final_regrets = [r["cumulative_regret"] for r in all_results
                        if r["depth"] == depth and r["T"] == args.T]
        mean_r = np.mean(final_regrets)
        std_r = np.std(final_regrets)
        print(f"d={depth:<6} {depth*args.b:<8} {args.b**depth:<8} {mean_r:<15,.0f} {std_r:<15,.0f}")

    # Crossover analysis
    print(f"\nCROSSOVER ANALYSIS (mean regret by checkpoint):")
    checkpoints_T = sorted(set(r["T"] for r in all_results))
    print(f"{'T':<12}", end="")
    for depth in range(1, args.max_depth + 1):
        print(f"{'d='+str(depth):<15}", end="")
    print("best")

    crossover_found = False
    for cp_t in checkpoints_T:
        print(f"{cp_t:<12,}", end="")
        means = []
        for depth in range(1, args.max_depth + 1):
            regrets = [r["cumulative_regret"] for r in all_results
                      if r["depth"] == depth and r["T"] == cp_t]
            m = np.mean(regrets)
            means.append(m)
            print(f"{m:<15,.0f}", end="")
        best_d = np.argmin(means) + 1
        marker = " <-- CROSSOVER" if best_d > 1 and not crossover_found else ""
        if best_d > 1:
            crossover_found = True
        print(f"d={best_d}{marker}")

    if not crossover_found:
        print(f"\nNo crossover found by T={args.T:,}. d=1 remains optimal throughout.")

    # Save CSV
    output_path = args.output or f"/tmp/synthetic_crossover_T{args.T}_b{args.b}_D{args.D}.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth", "seed", "T", "cumulative_regret"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} rows to {output_path}")


if __name__ == "__main__":
    main()
