#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Compute oracle Δ_RQ and Δ_Lin for datasets via 5-fold CV.

Δ_RQ: fit RQ codebook on train folds, assign codes, compute per-centroid
per-class means on train, predict on held-out. MAE vs one-hot.

Δ_Lin: fit Ridge Regression on raw features to predict one-hot labels.
MAE on held-out.

Also computes Laplace-smoothed variant for Δ_RQ.

Output CSV: dataset,N,d,C,depth,delta_rq_mae,delta_rq_mae_smoothed,
            delta_lin_mae,delta_gap,delta_gap_smoothed,rho

Usage:
    buck run @mode/opt mitra/projects/fi_trunk_tail/scripts/experiments:oracle_delta -- \
        --datasets miniboone real_hepmass skin_segmentation \
        --output /tmp/oracle_delta_new.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from mitra.projects.fi_trunk_tail.modules.data import load_dataset
from mitra.projects.fi_trunk_tail.modules.features import collect_raw_features
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _unpack_rq_codes(
    rq: "faiss.ResidualQuantizer",
    packed_bytes: np.ndarray,
    n_samples: int,
    d_cut: int,
    nbits: int,
) -> np.ndarray:
    """Unpack FAISS packed binary codes to per-level integer codes."""
    mask = (1 << nbits) - 1
    packed_ints = np.zeros(n_samples, dtype=np.int64)
    for byte_idx in range(packed_bytes.shape[1]):
        packed_ints += packed_bytes[:, byte_idx].astype(np.int64) << (8 * byte_idx)
    all_level_codes = np.zeros((n_samples, d_cut), dtype=np.int64)
    for level in range(d_cut):
        all_level_codes[:, level] = (packed_ints >> (level * nbits)) & mask
    return all_level_codes


def compute_delta_rq_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    d_cut: int,
    nbits: int = 4,
    eta: float = 0.5,
) -> tuple[float, float]:
    """Compute Δ_RQ (unsmoothed and smoothed) for one fold at one depth.

    Uses gradient-boosted additive model with η damping:
      Level 0: fit per-centroid means on raw one-hot labels
      Level ℓ: fit per-centroid means on residual targets
        target_ℓ = one_hot - clip(η^0 · f_0 + η^1 · f_1 + ... + η^{ℓ-1} · f_{ℓ-1})
      Prediction: clip(Σ_{ℓ} η^ℓ · f_ℓ(k_ℓ, a))

    This matches the actual Counter-RQ bandit's gradient-boosting structure.

    Returns (mae_unsmoothed, mae_smoothed).
    """
    import faiss

    dim = X_train.shape[1]
    b = 1 << nbits  # centroids per level

    # Train RQ on train fold
    rq = faiss.ResidualQuantizer(dim, d_cut, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.verbose = False
    rq.train(X_train.astype(np.float32))

    # Assign codes to train and test
    train_packed = rq.compute_codes(X_train.astype(np.float32))
    train_codes = _unpack_rq_codes(rq, train_packed, len(X_train), d_cut, nbits)
    test_packed = rq.compute_codes(X_test.astype(np.float32))
    test_codes = _unpack_rq_codes(rq, test_packed, len(X_test), d_cut, nbits)

    n_train = len(X_train)
    n_test = len(X_test)

    # One-hot labels for train
    train_onehot = np.zeros((n_train, n_classes), dtype=np.float64)
    for i in range(n_train):
        train_onehot[i, y_train[i]] = 1.0

    # Fit levels sequentially: each level targets the residual
    level_means = np.zeros((d_cut, b, n_classes), dtype=np.float64)
    level_means_smooth = np.zeros((d_cut, b, n_classes), dtype=np.float64)

    # Running cumulative prediction on train (for computing residual targets)
    train_cumul = np.zeros((n_train, n_classes), dtype=np.float64)

    for level in range(d_cut):
        # Residual target: one_hot - clip(cumulative prediction so far)
        residual_target = train_onehot - np.clip(train_cumul, 0.0, 1.0)

        # Per-centroid mean of residual targets
        centroid_sum = np.zeros((b, n_classes), dtype=np.float64)
        centroid_count = np.zeros(b, dtype=np.float64)
        for i in range(n_train):
            code = train_codes[i, level]
            centroid_sum[code] += residual_target[i]
            centroid_count[code] += 1.0

        for code in range(b):
            if centroid_count[code] > 0:
                level_means[level, code] = centroid_sum[code] / centroid_count[code]
            level_means_smooth[level, code] = (
                (centroid_sum[code] + 1.0 / n_classes)
                / (centroid_count[code] + 1.0)
            )

        # Update cumulative prediction with η damping
        eta_l = eta ** level
        for i in range(n_train):
            code = train_codes[i, level]
            train_cumul[i] += eta_l * level_means[level, code]

    # Predict on test: clip(Σ_ℓ η^ℓ · f_ℓ(k_ℓ))
    mae_sum = 0.0
    mae_sum_smoothed = 0.0
    for i in range(n_test):
        pred = np.zeros(n_classes, dtype=np.float64)
        pred_smooth = np.zeros(n_classes, dtype=np.float64)
        for level in range(d_cut):
            eta_l = eta ** level
            code = test_codes[i, level]
            pred += eta_l * level_means[level, code]
            pred_smooth += eta_l * level_means_smooth[level, code]
        pred = np.clip(pred, 0.0, 1.0)
        pred_smooth = np.clip(pred_smooth, 0.0, 1.0)

        one_hot = np.zeros(n_classes, dtype=np.float64)
        one_hot[y_test[i]] = 1.0

        mae_sum += np.abs(pred - one_hot).mean()
        mae_sum_smoothed += np.abs(pred_smooth - one_hot).mean()

    return mae_sum / n_test, mae_sum_smoothed / n_test


def compute_delta_lin_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> float:
    """Compute Δ_Lin for one fold. Returns MAE."""
    # One-hot encode labels
    Y_train = np.zeros((len(y_train), n_classes), dtype=np.float64)
    for i, c in enumerate(y_train):
        Y_train[i, c] = 1.0

    # Fit Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)

    # Predict on test
    Y_pred = ridge.predict(X_test)
    # Clip to [0, 1] and renormalize
    Y_pred = np.clip(Y_pred, 0.0, 1.0)
    row_sums = Y_pred.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    Y_pred = Y_pred / row_sums

    # MAE vs one-hot
    mae_sum = 0.0
    for i in range(len(X_test)):
        one_hot = np.zeros(n_classes, dtype=np.float64)
        one_hot[y_test[i]] = 1.0
        mae_sum += np.abs(Y_pred[i] - one_hot).mean()

    return mae_sum / len(X_test)


def compute_rho(X: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation ratio (eta-squared) between features and labels."""
    n_classes = int(y.max()) + 1
    grand_mean = X.mean(axis=0)
    ss_between = 0.0
    ss_total = np.sum((X - grand_mean) ** 2)
    for c in range(n_classes):
        mask = y == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        class_mean = X[mask].mean(axis=0)
        ss_between += n_c * np.sum((class_mean - grand_mean) ** 2)
    if ss_total < 1e-15:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def compute_oracle_delta(
    dataset_name: str,
    depths: list[int],
    nbits: int = 4,
    n_folds: int = 5,
) -> list[dict[str, object]]:
    """Compute oracle deltas for a dataset at multiple depths."""
    print(f"Loading dataset: {dataset_name}", flush=True)

    # Load and merge all splits (same as runner)
    _train_ds, _val_ds, _test_ds, metadata = load_dataset(dataset_name)

    from mitra.projects.fi_trunk_tail.modules.data import TabularDataset as _TDS

    _all_cont = [
        ds.continuous for ds in [_train_ds, _val_ds, _test_ds]
        if ds.continuous is not None
    ]
    _all_cat = [
        ds.categorical for ds in [_train_ds, _val_ds, _test_ds]
        if ds.categorical is not None
    ]
    _all_labels = [ds.labels for ds in [_train_ds, _val_ds, _test_ds]]

    merged = _TDS(
        continuous=torch.cat(_all_cont).numpy() if _all_cont else None,
        categorical=torch.cat(_all_cat).numpy() if _all_cat else None,
        labels=torch.cat(_all_labels).numpy(),
    )

    # Collect raw features (scaled continuous + one-hot categorical)
    features_np, labels_np = collect_raw_features(merged)
    features_np = features_np.astype(np.float32)
    labels_np = labels_np.astype(np.int64)

    N = len(features_np)
    d = features_np.shape[1]
    C = metadata.n_classes
    print(f"  N={N}, d={d}, C={C}", flush=True)
    assert C > 0, f"Expected classification dataset, got n_classes={C}"

    # Compute rho (correlation ratio)
    rho = compute_rho(features_np, labels_np)
    print(f"  rho={rho:.4f}", flush=True)

    # Compute Δ_Lin in-sample (train and predict on all data)
    delta_lin = compute_delta_lin_fold(
        features_np, labels_np, features_np, labels_np, C,
    )
    print(f"  Δ_Lin={delta_lin:.6f} (in-sample)", flush=True)

    rows = []
    for depth in depths:
        print(f"  depth={depth}...", end="", flush=True)
        # In-sample: train and predict on same data
        delta_rq, delta_rq_smooth = compute_delta_rq_fold(
            features_np, labels_np, features_np, labels_np,
            C, depth, nbits,
        )
        delta_gap = delta_lin - delta_rq
        delta_gap_smooth = delta_lin - delta_rq_smooth

        print(f" Δ_RQ={delta_rq:.6f}, gap={delta_gap:.6f}", flush=True)

        rows.append({
            "dataset": dataset_name,
            "N": N,
            "d": d,
            "C": C,
            "depth": depth,
            "delta_rq_mae": delta_rq,
            "delta_rq_mae_smoothed": delta_rq_smooth,
            "delta_lin_mae": delta_lin,
            "delta_gap": delta_gap,
            "delta_gap_smoothed": delta_gap_smooth,
            "rho": rho,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute oracle Δ_RQ and Δ_Lin")
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Datasets to compute deltas for",
    )
    parser.add_argument(
        "--depths", nargs="+", type=int, default=[1, 2, 3, 4],
        help="RQ depths to evaluate",
    )
    parser.add_argument("--nbits", type=int, default=4, help="Bits per RQ level")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument(
        "--output", type=str, default="/tmp/oracle_delta_new.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--append-to", type=str, default=None,
        help="Existing CSV to append to (copies existing rows + new rows to --output)",
    )
    args = parser.parse_args()

    all_rows = []

    # Load existing CSV if appending
    if args.append_to and Path(args.append_to).exists():
        with open(args.append_to) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
        print(f"Loaded {len(all_rows)} existing rows from {args.append_to}", flush=True)

    for ds in args.datasets:
        rows = compute_oracle_delta(ds, args.depths, args.nbits, args.n_folds)
        all_rows.extend(rows)

    # Write output
    cols = [
        "dataset", "N", "d", "C", "depth",
        "delta_rq_mae", "delta_rq_mae_smoothed", "delta_lin_mae",
        "delta_gap", "delta_gap_smoothed", "rho",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"\nWrote {len(all_rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
