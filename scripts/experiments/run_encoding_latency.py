#!/usr/bin/env python3


"""Encoding latency benchmark: flat codebook vs RQ tree.

Generates 10^6 synthetic vectors in d=256 using make_blobs, then measures
per-query encoding latency as the target vocabulary size scales from 2^4 to
2^16.

Flat codebook: brute-force nearest-centroid search, O(d * V).
RQ tree (b=16): residual quantization with 16 centroids per level, O(d * L * b).

At V = 2^16, theory predicts a ~1024x gap. This proves the hierarchical
bit-depths are a structural necessity at scale, and the encoding step doesn't
bottleneck the O(1) bandit decision.

Example:
    python3 scripts/experiments/run_encoding_latency.py -- \
        --n-vectors 1000000 --dim 256 --branching-factor 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np
from sklearn.datasets import make_blobs


@dataclass
class EncodingLatencyConfig:
    """Configuration for encoding latency benchmark."""

    n_vectors: int = 1_000_000
    dim: int = 256
    branching_factor: int = 16  # b=16 centroids per RQ level → nbits=4
    vocab_powers: list[int] = field(
        default_factory=lambda: [4, 6, 8, 10, 12, 14, 16]
    )
    n_queries: int = 10_000
    n_warmup: int = 200
    seed: int = 42


def _nbits_from_branching(b: int) -> int:
    """Convert branching factor to nbits: b=16 → nbits=4."""
    nbits = int(math.log2(b))
    assert 2**nbits == b, f"Branching factor must be power of 2, got {b}"
    return nbits


def benchmark_vocab_size(
    vocab_power: int,
    data: np.ndarray,
    queries: np.ndarray,
    config: EncodingLatencyConfig,
) -> dict[str, object]:
    """Benchmark flat vs RQ encoding at a given vocabulary size."""
    V = 2**vocab_power
    d = config.dim
    nbits = _nbits_from_branching(config.branching_factor)
    # Levels needed so that b^L >= V: L = ceil(vocab_power / nbits)
    L = max(1, math.ceil(vocab_power / nbits))

    # Use enough training data: at least 40 * V, capped by available data
    n_train = min(len(data), max(50_000, V * 40))
    train_data = data[:n_train].copy()

    # =========================================================================
    # Flat codebook: IndexFlatL2 with V centroids
    # =========================================================================
    # Use random sample from data as centroids (latency is independent of
    # centroid quality — purely a function of V and d).
    rng = np.random.RandomState(config.seed)
    centroid_idx = rng.choice(len(data), size=min(V, len(data)), replace=False)
    centroids = data[centroid_idx].copy()

    flat_index = faiss.IndexFlatL2(d)
    flat_index.add(centroids)

    # Warmup
    flat_index.search(queries[: config.n_warmup], 1)

    # Benchmark
    t0 = time.perf_counter()
    flat_index.search(queries, 1)
    flat_time = time.perf_counter() - t0
    flat_us = (flat_time / len(queries)) * 1e6

    # =========================================================================
    # RQ tree: faiss.ResidualQuantizer with L levels, nbits bits per level
    # =========================================================================
    rq = faiss.ResidualQuantizer(d, L, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = False
    rq.train(train_data)

    # Warmup
    rq.compute_codes(queries[: config.n_warmup])

    # Benchmark
    t0 = time.perf_counter()
    rq.compute_codes(queries)
    rq_time = time.perf_counter() - t0
    rq_us = (rq_time / len(queries)) * 1e6

    speedup = flat_us / rq_us if rq_us > 0 else float("inf")
    theory_flat = d * V
    theory_rq = d * L * config.branching_factor
    theory_speedup = theory_flat / theory_rq if theory_rq > 0 else float("inf")

    return {
        "vocab_power": vocab_power,
        "vocab_size": V,
        "n_levels": L,
        "flat_us_per_query": round(flat_us, 3),
        "rq_us_per_query": round(rq_us, 3),
        "speedup": round(speedup, 2),
        "theory_flat_ops": theory_flat,
        "theory_rq_ops": theory_rq,
        "theory_speedup": round(theory_speedup, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encoding latency: flat codebook vs RQ tree"
    )
    parser.add_argument("--n-vectors", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--branching-factor", type=int, default=16, help="b: centroids per RQ level")
    parser.add_argument(
        "--vocab-powers",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 14, 16],
        help="Target vocab sizes as powers of 2 (default: 4..16)",
    )
    parser.add_argument("--n-queries", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default="/tmp/encoding_latency")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = EncodingLatencyConfig(
        n_vectors=args.n_vectors,
        dim=args.dim,
        branching_factor=args.branching_factor,
        vocab_powers=args.vocab_powers,
        n_queries=args.n_queries,
        seed=args.seed,
    )

    nbits = _nbits_from_branching(config.branching_factor)

    print("=" * 80)
    print("ENCODING LATENCY BENCHMARK: Flat Codebook vs RQ Tree")
    print("=" * 80)
    print(f"Data:    {config.n_vectors:,} vectors, d={config.dim}")
    print(f"RQ:      b={config.branching_factor} (nbits={nbits})")
    print(f"Vocab:   {[2**p for p in config.vocab_powers]}")
    print(f"Queries: {config.n_queries:,}")
    print()

    # Generate synthetic data
    print("Generating synthetic data with make_blobs...")
    data, _ = make_blobs(
        n_samples=config.n_vectors,
        n_features=config.dim,
        centers=min(1000, config.n_vectors // 1000),
        random_state=config.seed,
    )
    data = data.astype(np.float32)
    queries = data[: config.n_queries].copy()
    print(f"Data shape: {data.shape}")
    print()

    # Benchmark each vocab size
    results: list[dict[str, object]] = []
    for vp in config.vocab_powers:
        V = 2**vp
        L = max(1, math.ceil(vp / nbits))
        print(f"V=2^{vp}={V:>6,d}  (RQ: {L} levels x {config.branching_factor} centroids)...")
        row = benchmark_vocab_size(vp, data, queries, config)
        results.append(row)
        print(
            f"  Flat: {row['flat_us_per_query']:>8.2f} us/q   "
            f"RQ: {row['rq_us_per_query']:>8.2f} us/q   "
            f"Speedup: {row['speedup']:>7.1f}x  (theory: {row['theory_speedup']:.0f}x)"
        )
        print()

    # Summary table
    print()
    print("=" * 90)
    print(
        f"{'Vocab':>8s}  {'Levels':>6s}  "
        f"{'Flat (us/q)':>11s}  {'RQ (us/q)':>11s}  "
        f"{'Speedup':>8s}  {'Theory':>8s}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['vocab_size']:>8,d}  {r['n_levels']:>6d}  "
            f"{r['flat_us_per_query']:>11.2f}  {r['rq_us_per_query']:>11.2f}  "
            f"{r['speedup']:>7.1f}x  {r['theory_speedup']:>7.0f}x"
        )
    print("-" * 90)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "encoding_latency.json"
    output = {"results": results, "config": asdict(config)}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
