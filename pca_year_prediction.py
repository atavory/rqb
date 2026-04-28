# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""PCA + LinTS-DRQm on year_prediction.

Global PCA before codebook training:
1. Hold out 50K samples (unsupervised)
2. Fit PCA on holdout: R^90 -> R^k
3. Transform ALL samples to R^k
4. Train RQ codebook on holdout in R^k
5. Run standard LinTS-DRQm on remaining 450K in R^k

Also runs LinTS baseline (no RQ) on R^k for comparison.

Output: year_prediction_pca.csv on Manifold.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import subprocess
import time

import numpy as np
from sklearn.decomposition import PCA

from mitra.projects.fi_trunk_tail.modules.data import load_dataset
from mitra.projects.fi_trunk_tail.modules.features import collect_raw_features
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.codebook import (
    compute_residual_features,
    encode,
    train_rq_codebook,
)
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.lints import LinTSBaseline
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.lints_rq import LinTSDRQm

# --- Config ---
DATASET = "year_prediction"
RQ_N = 50_000
NBITS = 4
B = 1 << NBITS  # 16
D_CUT = 8  # max depth for DRQm (adaptive)
DRQM_MIN_LEVEL = 2
DRQM_MAX_LEVEL = 8
N_SEEDS = 30
NU = 0.1
ETA = 0.5
LAM = 1.0
PCA_K_VALUES = [8, 16]
CHECKPOINT_INTERVAL = 5000

CSV_PATH = os.path.expanduser("~/fi_trunk_tail_results/year_prediction_pca.csv")
MANIFOLD_PATH = (
    "manifold://fi_platform_ml_infra_fluent2_bucket/tree/"
    "fi_trunk_tail/year_prediction_pca.csv"
)
COLS = [
    "dataset",
    "method",
    "seed",
    "round",
    "d_cut",
    "regret",
    "rq_n",
    "cumulative_method_time",
]


def _load_features() -> tuple[np.ndarray, np.ndarray, int]:
    """Load year_prediction, merge splits, extract raw features."""
    train_ds, val_ds, test_ds, metadata = load_dataset(DATASET)
    # Merge all splits
    from torch.utils.data import ConcatDataset

    merged = ConcatDataset([train_ds, val_ds, test_ds])
    features, labels = collect_raw_features(merged)
    n_classes = metadata.n_classes
    assert features.shape[1] == 90, f"Expected d=90, got {features.shape[1]}"
    assert n_classes > 0, f"Expected n_classes > 0, got {n_classes}"
    print(
        f"Loaded {DATASET}: {features.shape[0]} samples, "
        f"d={features.shape[1]}, K={n_classes}"
    )
    return features, labels, n_classes


# Global shared data (set in main, read by workers after fork)
_shared_data: dict[str, object] = {}

# Lock for thread-safe CSV writes from forked workers
_csv_lock = mp.Lock()


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


def _run_single(args: tuple[int, int]) -> list[dict[str, str]]:
    """Run one (pca_k, seed) combination."""
    pca_k, seed = args
    t0_total = time.perf_counter()

    eval_feats_pca = _shared_data[f"eval_feats_pca_{pca_k}"]
    eval_codes = _shared_data[f"eval_codes_{pca_k}"]
    eval_residuals = _shared_data[f"eval_residuals_{pca_k}"]
    eval_labels = _shared_data["eval_labels"]
    n_classes = _shared_data["n_classes"]

    rng = np.random.RandomState(seed)
    T = len(eval_labels)
    order = rng.permutation(T)

    results = []
    method_name = f"LinTS-DRQm-PCA{pca_k}"

    # --- LinTS-DRQm on PCA features ---
    drqm = LinTSDRQm(
        feat_dim=pca_k,
        n_arms=n_classes,
        b_per_level=B,
        max_depth=D_CUT,
        min_level=DRQM_MIN_LEVEL,
        max_level=DRQM_MAX_LEVEL,
        nu=NU,
        lam=LAM,
        eta=ETA,
        seed=seed,
    )

    cumulative_regret = 0.0
    t_method = 0.0
    for t_idx in range(T):
        idx = order[t_idx]
        trunk = eval_codes[idx]
        resid = eval_residuals[idx]

        t0 = time.perf_counter()
        arm = drqm.select_arm(trunk, resid)
        reward = 1.0 if arm == eval_labels[idx] else 0.0
        cumulative_regret += 1.0 - reward
        drqm.update(trunk, resid, arm, reward)
        t_method += time.perf_counter() - t0

        if (t_idx + 1) % CHECKPOINT_INTERVAL == 0 or t_idx == T - 1:
            row = {
                "dataset": DATASET,
                "method": method_name,
                "seed": str(seed),
                "round": str(t_idx + 1),
                "d_cut": str(D_CUT),
                "regret": f"{cumulative_regret:.1f}",
                "rq_n": str(RQ_N),
                "cumulative_method_time": f"{t_method:.2f}",
            }
            results.append(row)
            _append_row(row)
            print(
                f"  {method_name} s={seed} t={t_idx+1}/{T} "
                f"regret={cumulative_regret:.1f} rate={cumulative_regret/(t_idx+1):.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0_total
    print(
        f"  {method_name} seed={seed} DONE: regret={cumulative_regret:.1f} "
        f"rate={cumulative_regret/T:.4f} time={elapsed:.1f}s",
        flush=True,
    )

    # --- LinTS baseline on PCA features ---
    lints = LinTSBaseline(
        input_dim=pca_k + 1,  # +1 for intercept
        n_arms=n_classes,
        lambda_prior=LAM,
        nu=NU,
        rng=np.random.RandomState(seed + 1000),
    )
    lints_method = f"LinTS-PCA{pca_k}"
    cumulative_regret = 0.0
    t_method = 0.0
    rng2 = np.random.RandomState(seed)
    order2 = rng2.permutation(T)

    for t_idx in range(T):
        idx = order2[t_idx]
        x = np.append(eval_feats_pca[idx], 1.0)  # add intercept

        t0 = time.perf_counter()
        arm = lints.select_arm(x)
        reward = 1.0 if arm == eval_labels[idx] else 0.0
        cumulative_regret += 1.0 - reward
        lints.update(x, arm, reward)
        t_method += time.perf_counter() - t0

        if (t_idx + 1) % CHECKPOINT_INTERVAL == 0 or t_idx == T - 1:
            row = {
                "dataset": DATASET,
                "method": lints_method,
                "seed": str(seed),
                "round": str(t_idx + 1),
                "d_cut": "0",
                "regret": f"{cumulative_regret:.1f}",
                "rq_n": str(RQ_N),
                "cumulative_method_time": f"{t_method:.2f}",
            }
            results.append(row)
            _append_row(row)
            print(
                f"  {lints_method} s={seed} t={t_idx+1}/{T} "
                f"regret={cumulative_regret:.1f} rate={cumulative_regret/(t_idx+1):.4f}",
                flush=True,
            )

    print(
        f"  {lints_method} seed={seed} DONE: regret={cumulative_regret:.1f} "
        f"rate={cumulative_regret/T:.4f}",
        flush=True,
    )

    return results


def main() -> None:
    features, labels, n_classes = _load_features()
    N = features.shape[0]
    d = features.shape[1]

    # Inductive split: holdout for PCA + codebook, eval for bandit
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    holdout_idx = perm[:RQ_N]
    eval_idx = perm[RQ_N:]

    holdout_feats = features[holdout_idx]
    eval_feats = features[eval_idx]
    eval_labels_arr = labels[eval_idx]

    print(
        f"Inductive split: {len(holdout_idx)} holdout, "
        f"{len(eval_idx)} eval, d={d}, K={n_classes}"
    )

    _shared_data["eval_labels"] = eval_labels_arr
    _shared_data["n_classes"] = n_classes

    # For each PCA k: fit PCA on holdout, transform all, train codebook, compute codes
    for pca_k in PCA_K_VALUES:
        print(f"\n--- PCA k={pca_k} ---")

        # 1. Fit PCA on holdout
        pca = PCA(n_components=pca_k)
        pca.fit(holdout_feats)
        explained = sum(pca.explained_variance_ratio_)
        print(f"  PCA: {d} -> {pca_k}, explained variance: {explained:.4f}")

        # 2. Transform all samples
        holdout_pca = pca.transform(holdout_feats).astype(np.float32)
        eval_pca = pca.transform(eval_feats).astype(np.float32)

        # 3. Train RQ codebook on holdout in R^k
        rq, centroids = train_rq_codebook(holdout_pca, d_cut=D_CUT, nbits=NBITS)
        print(f"  RQ codebook trained: {D_CUT} levels, {B} centroids/level, dim={pca_k}")

        # 4. Encode eval set
        codes = encode(rq, eval_pca, d_cut=D_CUT, nbits=NBITS)
        residuals = compute_residual_features(eval_pca, codes, centroids, d_cut=D_CUT)

        # Store in shared data for workers
        _shared_data[f"eval_feats_pca_{pca_k}"] = eval_pca
        _shared_data[f"eval_codes_{pca_k}"] = codes
        _shared_data[f"eval_residuals_{pca_k}"] = residuals

    # Build work list: all (pca_k, seed) combinations
    work = [(pca_k, seed) for pca_k in PCA_K_VALUES for seed in range(1, N_SEEDS + 1)]
    n_workers = min(len(work), 60)
    print(f"\nRunning {len(work)} jobs across {n_workers} workers")

    # Use fork context explicitly (spawn won't share module globals)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        all_results = pool.map(_run_single, work)

    all_rows = [row for result in all_results for row in result]

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in sorted(
            all_rows,
            key=lambda r: (r["method"], int(r["seed"]), int(r["round"])),
        ):
            w.writerow(r)
    print(f"\nWrote {len(all_rows)} rows to {CSV_PATH}")

    # Upload to Manifold
    try:
        subprocess.run(["manifold", "rm", MANIFOLD_PATH], capture_output=True, timeout=30)
        subprocess.run(
            ["manifold", "put", CSV_PATH, MANIFOLD_PATH],
            capture_output=True,
            timeout=120,
        )
        print(f"Uploaded to {MANIFOLD_PATH}")
    except Exception as e:
        print(f"Manifold upload failed: {e}")


if __name__ == "__main__":
    main()
