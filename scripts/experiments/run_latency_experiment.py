#!/usr/bin/env python3


"""Latency benchmark: bandit decision and update cost vs embedding dimension.

Benchmarks select_arm and update latency for each bandit method as the
embedding dimension grows. Uses synthetic data (random embeddings, fit RQ,
random rewards) so there is no dataset dependency.

Example:
    python3 scripts/experiments/run_latency_experiment.py -- \
        --embedding-dims 64 128 256 512 --n-arms 5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import logging
from modules.bandits import HierarchicalThompsonSampling
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class LinTSBaseline:
    """Linear Thompson Sampling (duplicated here for standalone benchmarking)."""

    def __init__(
        self, input_dim: int, n_arms: int, lambda_prior: float = 1.0, nu: float = 0.1
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.lambda_prior = lambda_prior
        self.nu = nu

        self.B: list[np.ndarray] = [
            lambda_prior * np.eye(input_dim) for _ in range(n_arms)
        ]
        self.f: list[np.ndarray] = [np.zeros(input_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        x = context.flatten()
        best_arm = 0
        best_val = -float("inf")
        for a in range(self.n_arms):
            B_inv = np.linalg.solve(self.B[a], np.eye(self.d))
            mu = B_inv @ self.f[a]
            theta = np.random.multivariate_normal(mu, self.nu**2 * B_inv)
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        x = context.flatten()
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x


class ProjectedRQLinTS:
    """LinTS on PCA-projected RQ trunk reconstructions."""

    def __init__(
        self,
        trunk_recon: np.ndarray,
        n_components: int,
        n_arms: int,
        lambda_prior: float = 1.0,
        nu: float = 0.1,
    ) -> None:
        self.pca = PCA(n_components=n_components)
        self.pca.fit(trunk_recon)
        self.lints = LinTSBaseline(n_components, n_arms, lambda_prior, nu)

    def select_arm(self, features: np.ndarray) -> int:
        projected = self.pca.transform(features)
        return self.lints.select_arm(projected)

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        projected = self.pca.transform(features)
        self.lints.update(projected, arm, reward)


@dataclass
class LatencyConfig:
    """Configuration for latency experiments."""

    embedding_dims: list[int]
    n_arms: int = 5
    n_warmup: int = 200
    n_bench: int = 1000
    n_rq_levels: int = 2
    nbits: int = 6
    pca_k: int = 16
    seed: int = 42


def _time_fn(fn: object, n_iter: int) -> float:
    """Time a callable over n_iter calls, return mean time in microseconds."""
    # pyre-ignore[29]: Not a function.
    fn()  # warmup JIT / cache
    start = time.perf_counter()
    for _ in range(n_iter):
        # pyre-ignore[29]: Not a function.
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / n_iter) * 1e6  # microseconds


def benchmark_dim(
    dim: int,
    config: LatencyConfig,
) -> dict[str, float]:
    """Benchmark all methods at a given embedding dimension."""
    import faiss

    rng = np.random.RandomState(config.seed)
    n_train = 2000

    # Generate synthetic data
    embeddings = rng.randn(n_train, dim).astype(np.float32)

    # Fit RQ
    rq = faiss.ResidualQuantizer(dim, config.n_rq_levels, config.nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = False
    rq.train(embeddings)

    codes_np = rq.compute_codes(embeddings)
    codes = torch.from_numpy(codes_np.astype(np.int64))
    trunk_recon = rq.decode(codes_np)

    n_codes = max(int(codes.max().item()) + 1, 2**config.nbits)

    # Create bandits
    hier_ts = HierarchicalThompsonSampling(
        n_arms=config.n_arms,
        n_codes=n_codes,
        d_cut=config.n_rq_levels,
    )
    lints = LinTSBaseline(input_dim=dim, n_arms=config.n_arms)
    rqlints = LinTSBaseline(input_dim=dim, n_arms=config.n_arms)
    proj = ProjectedRQLinTS(
        trunk_recon=trunk_recon,
        n_components=config.pca_k,
        n_arms=config.n_arms,
    )

    # Warmup: feed some data so posteriors are non-trivial
    for _i in range(config.n_warmup):
        idx = rng.randint(n_train)
        c = codes[idx : idx + 1]
        e = embeddings[idx : idx + 1]
        r = trunk_recon[idx : idx + 1]
        reward = float(rng.binomial(1, 0.5))

        arm = hier_ts.select_arm(c).item()
        hier_ts.update(c, torch.tensor([arm]), torch.tensor([reward]))

        arm = lints.select_arm(e)
        lints.update(e, arm, reward)

        arm = rqlints.select_arm(r)
        rqlints.update(r, arm, reward)

        arm = proj.select_arm(r)
        proj.update(r, arm, reward)

    # Benchmark select_arm
    idx = rng.randint(n_train)
    c = codes[idx : idx + 1]
    e = embeddings[idx : idx + 1]
    r = trunk_recon[idx : idx + 1]

    hier_select = _time_fn(lambda: hier_ts.select_arm(c), config.n_bench)
    lints_select = _time_fn(lambda: lints.select_arm(e), config.n_bench)
    rqlints_select = _time_fn(lambda: rqlints.select_arm(r), config.n_bench)
    proj_select = _time_fn(lambda: proj.select_arm(r), config.n_bench)

    # Benchmark update
    reward = 1.0

    def hier_update() -> None:
        arm = rng.randint(config.n_arms)
        hier_ts.update(c, torch.tensor([arm]), torch.tensor([reward]))

    def lints_update() -> None:
        arm = rng.randint(config.n_arms)
        lints.update(e, arm, reward)

    def rqlints_update() -> None:
        arm = rng.randint(config.n_arms)
        rqlints.update(r, arm, reward)

    def proj_update() -> None:
        arm = rng.randint(config.n_arms)
        proj.update(r, arm, reward)

    hier_upd = _time_fn(hier_update, config.n_bench)
    lints_upd = _time_fn(lints_update, config.n_bench)
    rqlints_upd = _time_fn(rqlints_update, config.n_bench)
    proj_upd = _time_fn(proj_update, config.n_bench)

    return {
        "dim": dim,
        "hier_select_us": hier_select,
        "lints_select_us": lints_select,
        "rqlints_select_us": rqlints_select,
        "proj_select_us": proj_select,
        "hier_update_us": hier_upd,
        "lints_update_us": lints_upd,
        "rqlints_update_us": rqlints_upd,
        "proj_update_us": proj_upd,
    }


def run_latency_experiment(config: LatencyConfig) -> list[dict[str, float]]:
    """Run latency benchmark across all embedding dimensions."""
    results: list[dict[str, float]] = []
    for dim in config.embedding_dims:
        logger.info(f"Benchmarking dim={dim}...")
        row = benchmark_dim(dim, config)
        results.append(row)
        logger.info(
            f"  select_arm (us): HierTS={row['hier_select_us']:.1f}, "
            f"LinTS={row['lints_select_us']:.1f}, "
            f"RQ-LinTS={row['rqlints_select_us']:.1f}, "
            f"ProjRQ-{config.pca_k}={row['proj_select_us']:.1f}"
        )
        logger.info(
            f"  update (us):     HierTS={row['hier_update_us']:.1f}, "
            f"LinTS={row['lints_update_us']:.1f}, "
            f"RQ-LinTS={row['rqlints_update_us']:.1f}, "
            f"ProjRQ-{config.pca_k}={row['proj_update_us']:.1f}"
        )

    # Print summary table
    logger.info("\n" + "=" * 100)
    logger.info(f"LATENCY RESULTS (n_arms={config.n_arms}, PCA k={config.pca_k})")
    logger.info("=" * 100)
    logger.info(
        f"{'dim':>5s}  {'HierTS sel':>12s}  {'LinTS sel':>12s}  "
        f"{'RQ-LinTS sel':>14s}  {'ProjRQ sel':>12s}  "
        f"{'LinTS/Proj':>11s}"
    )
    logger.info("-" * 100)
    for row in results:
        speedup = (
            row["lints_select_us"] / row["proj_select_us"]
            if row["proj_select_us"] > 0
            else float("inf")
        )
        logger.info(
            f"{row['dim']:5.0f}  {row['hier_select_us']:10.1f}us  "
            f"{row['lints_select_us']:10.1f}us  {row['rqlints_select_us']:12.1f}us  "
            f"{row['proj_select_us']:10.1f}us  {speedup:9.1f}x"
        )
    logger.info("-" * 100)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latency benchmark: bandit decision cost vs embedding dimension"
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Embedding dimensions to benchmark",
    )
    parser.add_argument("--n-arms", type=int, default=5, help="Number of bandit arms")
    parser.add_argument(
        "--n-warmup", type=int, default=200, help="Warmup iterations per method"
    )
    parser.add_argument(
        "--n-bench", type=int, default=1000, help="Benchmark iterations per method"
    )
    parser.add_argument(
        "--pca-k", type=int, default=16, help="PCA projection dimension"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/latency_results",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = LatencyConfig(
        embedding_dims=args.embedding_dims,
        n_arms=args.n_arms,
        n_warmup=args.n_warmup,
        n_bench=args.n_bench,
        pca_k=args.pca_k,
        seed=args.seed,
    )

    logger.info("=" * 60)
    logger.info("LATENCY EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Embedding dims: {config.embedding_dims}")
    logger.info(f"n_arms: {config.n_arms}, PCA k: {config.pca_k}")
    logger.info(f"n_warmup: {config.n_warmup}, n_bench: {config.n_bench}")

    start_time = time.time()
    results = run_latency_experiment(config)
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latency_results.json"
    output = {
        "results": results,
        "config": asdict(config),
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=float)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
