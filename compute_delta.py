#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Compute additive misspecification Δ_ℓ for TS-RQ.

For each (dataset, depth, seed), splits data into codebook/eval,
trains RQ, then computes Δ_ℓ = E_x[max_a |r*(x,a) - Σ_i f_i*(k_i(x),a)|]
where f_i* are the optimal scalar lookup tables (OLS on one-hot Φ).

Cross-validates over seeds to check stability.

Requirements: numpy scipy faiss-cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

_ALGS_DIR = Path(__file__).resolve().parent
if str(_ALGS_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGS_DIR))

from codebook import encode, train_rq_codebook


def compute_delta_scalar(
    codes_fit: np.ndarray,
    labels_fit: np.ndarray,
    codes_eval: np.ndarray,
    labels_eval: np.ndarray,
    n_arms: int,
    depth: int,
    b: int,
) -> float:
    """Compute Δ_ℓ for scalar base learner (TS-RQ).

    Fits optimal additive lookup tables on (codes_fit, labels_fit),
    evaluates on (codes_eval, labels_eval).

    Returns Δ_ℓ = E_x[max_a |r*(x,a) - Σ_i f_i*(k_i(x), a)|].
    """
    n_fit = len(codes_fit)
    n_eval = len(codes_eval)
    D = b * depth

    # Build sparse one-hot Φ matrix for fitting data
    rows = []
    cols = []
    for lvl in range(depth):
        rows.extend(range(n_fit))
        cols.extend(codes_fit[:, lvl].astype(int) + lvl * b)
    phi_fit = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_fit, D)
    )

    # OLS per arm: θ_a = (Φ^T Φ)^{-1} Φ^T r_a
    gram = (phi_fit.T @ phi_fit).toarray() + 1e-8 * np.eye(D)
    gram_inv = np.linalg.inv(gram)

    thetas = np.zeros((n_arms, D))
    for a in range(n_arms):
        r_a = (labels_fit == a).astype(np.float64)
        thetas[a] = gram_inv @ (phi_fit.T @ r_a)

    # Build sparse Φ for eval data
    rows_e = []
    cols_e = []
    for lvl in range(depth):
        rows_e.extend(range(n_eval))
        cols_e.extend(codes_eval[:, lvl].astype(int) + lvl * b)
    phi_eval = sparse.csr_matrix(
        (np.ones(len(rows_e)), (rows_e, cols_e)), shape=(n_eval, D)
    )

    # Compute max_a |r*(x,a) - θ_a^T Φ(x)| for each x
    max_abs_err = np.zeros(n_eval)
    for a in range(n_arms):
        r_star = (labels_eval == a).astype(np.float64)
        pred = phi_eval @ thetas[a]
        abs_err = np.abs(r_star - pred)
        max_abs_err = np.maximum(max_abs_err, abs_err)

    return float(np.mean(max_abs_err))


def compute_delta_for_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    max_depth: int = 4,
    nbits: int = 4,
    rq_n: int = 5000,
    n_seeds: int = 5,
    base_seed: int = 42,
) -> dict:
    """Compute Δ_ℓ for depths 1..max_depth, cross-validated over seeds."""
    n_total = len(features)
    n_arms = int(labels.max()) + 1
    b = 2 ** nbits

    results = {}
    for depth in range(1, max_depth + 1):
        deltas = []
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_total)

            cb_idx = perm[:rq_n]
            eval_idx = perm[rq_n:]

            # Train codebook on holdout
            rq, _centroids = train_rq_codebook(
                features[cb_idx], depth, nbits
            )

            # Encode both splits
            codes_fit = encode(rq, features[cb_idx], depth, nbits)
            codes_eval = encode(rq, features[eval_idx], depth, nbits)

            delta = compute_delta_scalar(
                codes_fit, labels[cb_idx],
                codes_eval, labels[eval_idx],
                n_arms, depth, b,
            )
            deltas.append(delta)

        results[depth] = {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "values": deltas,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute additive misspecification Δ_ℓ"
    )
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--rq-n", type=int, default=5000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Use miniboone for local testing
    from demo_miniboone import download_miniboone

    features, labels = download_miniboone()
    n_arms = int(labels.max()) + 1
    p_max = np.bincount(labels).max() / len(labels)
    print(f"MiniBooNE: N={len(features)}, d={features.shape[1]}, "
          f"K={n_arms}, p_max={p_max:.3f}")

    print(f"\nComputing Δ_ℓ (scalar/TS-RQ), {args.n_seeds} seeds:")
    print(f"{'depth':<8} {'Δ_ℓ mean':<12} {'Δ_ℓ std':<12} "
          f"{'2Δ_ℓ':<12} {'per-seed values'}")
    print("-" * 70)

    t0 = time.time()
    results = compute_delta_for_dataset(
        features, labels,
        max_depth=args.max_depth,
        nbits=args.nbits,
        rq_n=args.rq_n,
        n_seeds=args.n_seeds,
        base_seed=args.seed,
    )
    elapsed = time.time() - t0

    for depth in sorted(results.keys()):
        r = results[depth]
        vals = ", ".join(f"{v:.4f}" for v in r["values"])
        print(f"{depth:<8} {r['mean']:<12.4f} {r['std']:<12.4f} "
              f"{2*r['mean']:<12.4f} [{vals}]")

    print(f"\nDone in {elapsed:.1f}s")

    # Compare to observed R(T)/T if we have it
    print(f"\nFor reference: if R(T)/T converges to c·Δ_ℓ,")
    print(f"the constant c should be ≤ 2 (worst-case adversarial).")
    print(f"Comparing 2Δ_ℓ to observed regret rate would test the bound.")


if __name__ == "__main__":
    main()
