# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Synthetic data sufficiency experiment.

Generates synthetic bandit datasets with controlled d, K, N to sweep
N/(b·K·d²). Runs LinTS-DRQm vs LinTS on each. Outputs a CSV.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

from mitra.projects.fi_trunk_tail.scripts.experiments.algs.codebook import (
    compute_residual_features,
    encode,
    train_rq_codebook,
)
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.lints import LinTSBaseline
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.lints_rq import LinTSDRQm


SWEEP = [
    (5, 3),
    (5, 5),
    (5, 10),
    (8, 5),
    (10, 3),
    (10, 5),
    (10, 10),
    (15, 5),
    (15, 10),
    (20, 5),
]

N_TOTAL = 500_000
RQ_N = 50_000
NBITS = 4
B = 1 << NBITS  # 16
D_CUT = 1
N_SEEDS = 5
NU = 0.1
ETA = 0.5
LAM = 1.0

CSV_PATH = os.path.expanduser("~/fi_trunk_tail_results/synthetic_sufficiency.csv")
MANIFOLD_PATH = "manifold://fi_platform_ml_infra_fluent2_bucket/tree/fi_trunk_tail/synthetic_sufficiency.csv"
COLS = ["d", "K", "ratio", "method", "seed", "final_regret", "regret_rate"]


def generate_dataset(
    d: int, K: int, N: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    features = rng.randn(N, d).astype(np.float32)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    arm_weights = rng.randn(K, d).astype(np.float32) * 0.3
    scores = features @ arm_weights.T
    labels = scores.argmax(axis=1)
    return features, labels


def run_single(args: tuple[int, int, int]) -> list[dict[str, str]]:
    d, K, seed = args
    rng = np.random.RandomState(seed)
    features, labels = generate_dataset(d, K, N_TOTAL, rng)

    train_feats = features[:RQ_N]
    eval_feats = features[RQ_N:]
    eval_labels = labels[RQ_N:]
    T = len(eval_labels)

    ratio = N_TOTAL / (B * K * d * d)

    rq, centroids = train_rq_codebook(train_feats, d_cut=D_CUT, nbits=NBITS)
    codes = encode(rq, eval_feats, d_cut=D_CUT, nbits=NBITS)
    residuals = compute_residual_features(eval_feats, codes, centroids, d_cut=D_CUT)

    results = []

    # --- LinTS baseline ---
    lints = LinTSBaseline(
        input_dim=d, n_arms=K, lambda_prior=LAM, nu=NU,
        rng=np.random.RandomState(seed + 1000),
    )
    cumulative_regret = 0.0
    for t in range(T):
        x = eval_feats[t]
        arm = lints.select_arm(x)
        reward = 1.0 if arm == eval_labels[t] else 0.0
        cumulative_regret += 1.0 - reward
        lints.update(x, arm, reward)

    row_lints = {
        "d": str(d), "K": str(K), "ratio": f"{ratio:.4f}",
        "method": "LinTS", "seed": str(seed),
        "final_regret": f"{cumulative_regret:.1f}",
        "regret_rate": f"{cumulative_regret / T:.6f}",
    }
    results.append(row_lints)
    _append_row(row_lints)
    print(f"  LinTS d={d} K={K} seed={seed}: regret={cumulative_regret:.1f} rate={cumulative_regret/T:.4f}", flush=True)

    # --- LinTS-DRQm ---
    drqm = LinTSDRQm(
        feat_dim=d, n_arms=K, b_per_level=B,
        max_depth=D_CUT, min_level=1, max_level=D_CUT,
        nu=NU, lam=LAM, eta=ETA, seed=seed + 2000,
    )
    cumulative_regret = 0.0
    for t in range(T):
        trunk = codes[t]
        resid = residuals[t]
        arm = drqm.select_arm(trunk, resid)
        reward = 1.0 if arm == eval_labels[t] else 0.0
        cumulative_regret += 1.0 - reward
        drqm.update(trunk, resid, arm, reward)

    row_drqm = {
        "d": str(d), "K": str(K), "ratio": f"{ratio:.4f}",
        "method": "LinTS-DRQm", "seed": str(seed),
        "final_regret": f"{cumulative_regret:.1f}",
        "regret_rate": f"{cumulative_regret / T:.6f}",
    }
    results.append(row_drqm)
    _append_row(row_drqm)
    print(f"  DRQm  d={d} K={K} seed={seed}: regret={cumulative_regret:.1f} rate={cumulative_regret/T:.4f}", flush=True)

    return results


def _append_row(row: dict[str, str]) -> None:
    """Append a single row to CSV immediately (with file locking)."""
    import fcntl

    p = os.path.expanduser(CSV_PATH)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_header = not os.path.exists(p) or os.path.getsize(p) == 0
    with open(p, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def main() -> None:
    work = [(d, K, seed) for d, K in SWEEP for seed in range(1, N_SEEDS + 1)]
    n_workers = min(len(work), 30)
    print(f"Running {len(work)} jobs in parallel across {n_workers} workers")

    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        all_results = pool.map(run_single, work)

    all_rows = [row for result in all_results for row in result]
    print(f"\nAll done: {len(all_rows)} rows in {CSV_PATH}")

    try:
        subprocess.run(["manifold", "rm", MANIFOLD_PATH], capture_output=True, timeout=30)
        subprocess.run(["manifold", "put", CSV_PATH, MANIFOLD_PATH], capture_output=True, timeout=60)
        print(f"Uploaded to {MANIFOLD_PATH}")
    except Exception as e:
        print(f"Manifold upload failed: {e}")


if __name__ == "__main__":
    main()
