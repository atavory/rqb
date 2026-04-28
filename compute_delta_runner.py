#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Compute additive misspecification Δ_ℓ for all paper datasets.

For each (dataset, depth ℓ=1..5), trains RQ codebook and computes:
  Δ_ℓ^scalar = E_x[max_a |r*(x,a) - Σ_i f_i*(k_i(x),a)|]
where f_i* are optimal scalar lookup tables (OLS on one-hot Φ).

Cross-validates over 5 seeds (different codebook/eval splits).
Outputs CSV: dataset, depth, seed, delta_scalar, n_eval

Usage:
    python3 compute_delta_runner.py --data-dir /path/to/pt/files --output delta_results.csv
    python3 compute_delta_runner.py --data-dir /path/to/pt/files --datasets higgs covertype
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

# Ensure sibling imports work
_ALGS_DIR = Path(__file__).resolve().parent
if str(_ALGS_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGS_DIR))

from codebook import encode, train_rq_codebook

PAPER_DATASETS = [
    "airlines_delay",
    "bng_elevators",
    "bng_letter",
    "covertype",
    "beer_reviews",
    "hepmass",
    "higgs",
    "kddcup99",
    "miniboone",
    "poker_hand",
    "skin_segmentation",
    "susy",
    "year_prediction",
]


def load_dataset(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from .pt or .npz file."""
    import torch

    pt_file = data_dir / f"{name}.pt"
    npz_file = data_dir / f"{name}.npz"

    if pt_file.exists():
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            features = data["features"].numpy().astype(np.float32)
            labels = data["labels"].numpy().astype(np.int64)
        else:
            features = data[0].numpy().astype(np.float32)
            labels = data[1].numpy().astype(np.int64)
    elif npz_file.exists():
        data = np.load(npz_file)
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
    else:
        raise FileNotFoundError(f"No .pt or .npz file for {name} in {data_dir}")

    # Standardize
    mu = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    features = (features - mu) / std

    return features, labels


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Δ_ℓ for paper datasets"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing dataset .pt files",
    )
    parser.add_argument(
        "--output", type=str, default="delta_results.csv",
        help="Output CSV path (default: delta_results.csv)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Subset of datasets (default: all 13 paper datasets)",
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--rq-n", type=int, default=50000,
                        help="Codebook training holdout (default: 50000)")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    datasets = args.datasets or PAPER_DATASETS
    b = 2 ** args.nbits

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "depth", "seed", "delta_scalar",
            "n_arms", "n_fit", "n_eval",
        ])

        for ds_name in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name}")
            try:
                features, labels = load_dataset(data_dir, ds_name)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            n_total = len(features)
            n_arms = int(labels.max()) + 1
            p_max = np.bincount(labels).max() / n_total
            print(f"  N={n_total}, d={features.shape[1]}, K={n_arms}, "
                  f"p_max={p_max:.3f}")

            for depth in range(1, args.max_depth + 1):
                deltas = []
                for seed_idx in range(args.n_seeds):
                    seed = args.seed + seed_idx
                    rng = np.random.RandomState(seed)
                    perm = rng.permutation(n_total)

                    rq_n = min(args.rq_n, n_total // 2)
                    cb_idx = perm[:rq_n]
                    eval_idx = perm[rq_n:]

                    t0 = time.time()
                    rq, _centroids = train_rq_codebook(
                        features[cb_idx], depth, args.nbits
                    )
                    codes_fit = encode(rq, features[cb_idx], depth, args.nbits)
                    codes_eval = encode(
                        rq, features[eval_idx], depth, args.nbits
                    )

                    delta = compute_delta_scalar(
                        codes_fit, labels[cb_idx],
                        codes_eval, labels[eval_idx],
                        n_arms, depth, b,
                    )
                    deltas.append(delta)
                    elapsed = time.time() - t0

                    writer.writerow([
                        ds_name, depth, seed, f"{delta:.6f}",
                        n_arms, len(cb_idx), len(eval_idx),
                    ])
                    f.flush()

                mean_d = np.mean(deltas)
                std_d = np.std(deltas)
                print(f"  depth={depth}  Δ={mean_d:.4f} ± {std_d:.4f}  "
                      f"2Δ={2*mean_d:.4f}  ({elapsed:.1f}s/seed)")

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
