#!/usr/bin/env python3
"""Compute RQ reconstruction error at each depth for the unsupervised diagnostic.

For each dataset, runs residual quantization (iterative k-means) at depths 1..max_depth
and reports E[||x - x̂_d||²] at each depth. This is the bias proxy for the proposed
unsupervised diagnostic:

    d* = argmin_d { E[||x - x̂_d||²] + λ * d * b * A / T }

Usage:
    python3 scripts/diagnostic_quant_error.py
    python3 scripts/diagnostic_quant_error.py --datasets adult covertype --b 4 --max-depth 5
"""

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DatasetInfo:
    openml_id: int
    n_classes: int


DATASETS = {
    "adult": DatasetInfo(openml_id=1590, n_classes=2),
    "bank_marketing": DatasetInfo(openml_id=1461, n_classes=2),
    "jannis": DatasetInfo(openml_id=41168, n_classes=4),
    "volkert": DatasetInfo(openml_id=41166, n_classes=10),
    "covertype": DatasetInfo(openml_id=44121, n_classes=7),
    "letter": DatasetInfo(openml_id=6, n_classes=26),
    "helena": DatasetInfo(openml_id=41169, n_classes=100),
}


def load_dataset(name: str, data_home: str = "/tmp/sklearn_data") -> tuple[np.ndarray, int, int]:
    """Load dataset, return (features, n_classes, n_samples)."""
    info = DATASETS[name]
    print(f"  Loading {name} (OpenML {info.openml_id})...", end="", flush=True)
    data = fetch_openml(data_id=info.openml_id, as_frame=True, parser="auto", data_home=data_home)
    df = data.frame

    # Drop non-numeric columns (categoricals we can't RQ on)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = data.target.name if hasattr(data.target, "name") else "target"
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].values.astype(np.float32)

    # Drop rows with NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]

    print(f" {X.shape[0]} samples, {X.shape[1]} features, {info.n_classes} classes")
    return X, info.n_classes, X.shape[0]


def run_rq(X: np.ndarray, b: int, max_depth: int, n_seeds: int = 3) -> list[float]:
    """Run residual quantization and return mean reconstruction error at each depth.

    Returns list of length max_depth where entry[d-1] = E[||x - x̂_d||²].
    Averages over n_seeds k-means initializations.
    """
    n, d_feat = X.shape
    errors_by_depth = []

    for seed in range(n_seeds):
        residual = X.copy()
        reconstruction = np.zeros_like(X)

        seed_errors = []
        for depth in range(1, max_depth + 1):
            km = KMeans(n_clusters=b, n_init=1, random_state=seed * 100 + depth)
            assignments = km.fit_predict(residual)
            centroids = km.cluster_centers_

            # Update reconstruction
            reconstruction += centroids[assignments]
            residual = X - reconstruction

            # Reconstruction error = mean squared residual norm
            mse = np.mean(np.sum(residual ** 2, axis=1))
            seed_errors.append(mse)

        errors_by_depth.append(seed_errors)

    # Average over seeds
    return np.mean(errors_by_depth, axis=0).tolist()


def compute_diagnostic(
    quant_errors: list[float],
    b: int,
    n_classes: int,
    T: int,
    lam: float,
) -> tuple[int, list[float]]:
    """Compute the unsupervised diagnostic score at each depth.

    score(d) = E[||x - x̂_d||²] + λ * d * b * A / T

    Returns (d_star, scores) where d_star is the argmin (1-indexed).
    """
    scores = []
    for d_idx, qe in enumerate(quant_errors):
        d = d_idx + 1
        penalty = lam * d * b * n_classes / T
        scores.append(qe + penalty)

    d_star = int(np.argmin(scores)) + 1
    return d_star, scores


def main():
    parser = argparse.ArgumentParser(description="RQ reconstruction error diagnostic")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--b", type=int, default=4, help="Branching factor (codebook size per level)")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum RQ depth")
    parser.add_argument("--T", type=int, default=100_000, help="Bandit horizon")
    parser.add_argument("--seeds", type=int, default=3, help="K-means seeds to average")
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
        help="Lambda values to test",
    )
    args = parser.parse_args()

    print(f"Config: b={args.b}, max_depth={args.max_depth}, T={args.T}, seeds={args.seeds}")
    print(f"Lambdas: {args.lambdas}")
    print()

    all_results = {}

    for name in args.datasets:
        print(f"=== {name} ===")
        X, n_classes, n_samples = load_dataset(name)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)

        # Baseline: total variance (error at depth 0 = no quantization)
        total_var = np.mean(np.sum(X_scaled ** 2, axis=1))
        print(f"  Total variance (depth=0): {total_var:.2f}")

        # Run RQ
        print(f"  Running RQ (b={args.b}, depths 1..{args.max_depth}, {args.seeds} seeds)...", end="", flush=True)
        quant_errors = run_rq(X_scaled, args.b, args.max_depth, args.seeds)
        print(" done")

        # Print reconstruction errors
        print(f"  {'depth':<6} {'E[||r||²]':<12} {'% var explained':<18} {'marginal reduction'}")
        prev = total_var
        for d_idx, qe in enumerate(quant_errors):
            d = d_idx + 1
            pct = 100.0 * (1.0 - qe / total_var)
            marginal = prev - qe
            print(f"  d={d:<4} {qe:<12.4f} {pct:<18.1f}% {marginal:.4f}")
            prev = qe

        # Diagnostic scores for various lambdas
        print()
        header = f"  {'λ':<8}"
        for d in range(1, args.max_depth + 1):
            header += f"{'d=' + str(d):<12}"
        header += "d*"
        print(header)

        for lam in args.lambdas:
            d_star, scores = compute_diagnostic(quant_errors, args.b, n_classes, args.T, lam)
            row = f"  {lam:<8.1f}"
            for s in scores:
                row += f"{s:<12.4f}"
            row += f"{d_star}"
            print(row)

        all_results[name] = {
            "n_samples": n_samples,
            "n_features": X_scaled.shape[1],
            "n_classes": n_classes,
            "total_variance": total_var,
            "quant_errors": quant_errors,
        }
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY: Reconstruction error by depth")
    print(f"{'dataset':<16} {'D':<5} {'C':<5} {'N':<8}", end="")
    for d in range(1, args.max_depth + 1):
        print(f"{'d=' + str(d):<10}", end="")
    print()

    for name, r in all_results.items():
        print(f"{name:<16} {r['n_features']:<5} {r['n_classes']:<5} {r['n_samples']:<8}", end="")
        for qe in r["quant_errors"]:
            print(f"{qe:<10.3f}", end="")
        print()


if __name__ == "__main__":
    main()
