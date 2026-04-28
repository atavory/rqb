#!/usr/bin/env python3
"""End-to-end demo: contextual bandits on MiniBooNE.

Downloads the MiniBooNE dataset from UCI (cached), trains an RQ codebook,
and runs all 6 methods (3 baselines + 3 RQ variants) for multiple seeds.

Requirements (pip install):
    numpy scipy faiss-cpu

Usage:
    python demo_miniboone.py
    python demo_miniboone.py --n-runs 5
    python demo_miniboone.py --max-rounds 10000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Standalone imports: everything from sibling files via relative path
# ---------------------------------------------------------------------------
_ALGS_DIR = Path(__file__).resolve().parent
if str(_ALGS_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGS_DIR))

from codebook import compute_residual_features, encode, train_rq_codebook
from counter_rq import CounterDRQm
from lints import LinTSBaseline
from lints_rq import LinTSDRQm
from sgd_lints import SGDLinTS
from sgd_lints_rq import SGDLinDRQm


# ---------------------------------------------------------------------------
# Lightweight NIG TS (numpy only, no torch)
# ---------------------------------------------------------------------------

class SimpleTS:
    """Context-free Thompson Sampling via Normal-Inverse-Gamma posterior.

    Numpy-only replacement for NIGStats (which requires torch).
    """

    def __init__(self, n_arms: int, mu0: float = 0.0, lam: float = 1.0,
                 alpha: float = 1.0, beta: float = 1.0) -> None:
        self.n_arms = n_arms
        self.mu0 = np.full(n_arms, mu0)
        self.lam = np.full(n_arms, lam)
        self.alpha = np.full(n_arms, alpha)
        self.beta = np.full(n_arms, beta)

    def sample_means(self, rng: np.random.RandomState) -> np.ndarray:
        # variance ~ InvGamma(alpha, beta) = 1/Gamma(alpha, 1/beta)
        variance = 1.0 / rng.gamma(self.alpha, 1.0 / self.beta)
        variance = np.maximum(variance, 1e-10)
        std = np.sqrt(variance / self.lam)
        return rng.normal(self.mu0, std)

    def update(self, arm: int, obs: float) -> None:
        lam_new = self.lam[arm] + 1.0
        self.mu0[arm] = (self.lam[arm] * self.mu0[arm] + obs) / lam_new
        self.alpha[arm] += 0.5
        self.beta[arm] += 0.5 * self.lam[arm] * (obs - self.mu0[arm]) ** 2 / lam_new
        self.lam[arm] = lam_new


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

MINIBOONE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
CACHE_DIR = Path(os.environ.get("DRQ_CACHE", "/tmp/drq_demo_cache"))


def download_miniboone() -> tuple[np.ndarray, np.ndarray]:
    """Download MiniBooNE from UCI. Cached on disk.

    Returns:
        features: (N, 50) float32, standardized.
        labels: (N,) int64, values in {0, 1}.
    """
    cache_file = CACHE_DIR / "miniboone.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"Loaded cached MiniBooNE from {cache_file}", file=sys.stderr)
        return data["features"], data["labels"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = CACHE_DIR / "MiniBooNE_PID.txt"

    if not raw_file.exists():
        print("Downloading MiniBooNE from UCI...", file=sys.stderr)
        urllib.request.urlretrieve(MINIBOONE_URL, raw_file)
        print(f"  Saved to {raw_file}", file=sys.stderr)

    with open(raw_file) as f:
        header = f.readline().strip().split()
        n_signal, n_background = int(header[0]), int(header[1])

    raw = np.loadtxt(raw_file, skiprows=1)
    n_total = n_signal + n_background
    assert raw.shape[0] == n_total, f"Expected {n_total}, got {raw.shape[0]}"

    labels = np.zeros(n_total, dtype=np.int64)
    labels[:n_signal] = 1
    labels[n_signal:] = 0
    features = raw.astype(np.float32)

    mu = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    features = (features - mu) / std

    rng = np.random.RandomState(0)
    perm = rng.permutation(n_total)
    features, labels = features[perm], labels[perm]

    np.savez_compressed(cache_file, features=features, labels=labels)
    print(f"  Cached to {cache_file}", file=sys.stderr)
    return features, labels


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(
    rq_n: int = 5000,
    d_cut: int = 4,
    nbits: int = 4,
    n_runs: int = 3,
    seed: int = 42,
    max_rounds: int = 0,
    promotion_test: str = "ttest",
) -> None:
    features, labels = download_miniboone()
    n_total = len(features)
    n_arms = int(labels.max()) + 1
    n_features = features.shape[1]
    b_per_level = 2 ** nbits

    print(f"\nMiniBooNE: N={n_total}, d={n_features}, K={n_arms}")
    print(f"  p_max={np.bincount(labels).max() / n_total:.3f}")

    # Inductive split
    rng_split = np.random.RandomState(0)
    perm = rng_split.permutation(n_total)
    codebook_idx = perm[:rq_n]
    eval_idx = perm[rq_n:]
    n_rounds = len(eval_idx)
    if max_rounds > 0:
        n_rounds = min(n_rounds, max_rounds)
        eval_idx = eval_idx[:n_rounds]
    print(f"  Codebook: {rq_n}, Eval: {n_rounds}, depth={d_cut}, b={b_per_level}")

    # Train RQ codebook
    print("Training RQ codebook...", end="", flush=True)
    t0 = time.time()
    rq, centroids = train_rq_codebook(features[codebook_idx], d_cut, nbits)
    print(f" done ({time.time()-t0:.1f}s)")

    # Encode
    print("Encoding eval set...", end="", flush=True)
    t0 = time.time()
    eval_features = features[eval_idx]
    eval_labels = labels[eval_idx]
    eval_codes = encode(rq, eval_features, d_cut, nbits)
    residual_features = compute_residual_features(
        eval_features, eval_codes, centroids, d_cut
    )
    print(f" done ({time.time()-t0:.1f}s)")

    feature_dim = n_features + 1  # +1 for intercept

    # Run
    all_regrets: dict[str, list[float]] = {}
    all_times: dict[str, list[float]] = {}

    print(f"\nRunning {n_runs} seeds, T={n_rounds}:")
    for run_idx in range(n_runs):
        bandit_seed = seed + run_idx
        rng = np.random.RandomState(bandit_seed)
        order = rng.permutation(n_rounds)

        ts = SimpleTS(n_arms=n_arms)
        ts_rng = np.random.RandomState(bandit_seed)
        lints = LinTSBaseline(
            input_dim=feature_dim, n_arms=n_arms,
            lambda_prior=1.0, nu=0.1,
            rng=np.random.RandomState(bandit_seed),
        )
        sgd = SGDLinTS(
            input_dim=feature_dim, n_arms=n_arms,
            nu=0.1, lam=1.0, seed=bandit_seed,
        )
        counter_drqm = CounterDRQm(
            n_arms=n_arms, b_per_level=b_per_level, max_depth=d_cut,
            min_level=2, max_level=8, nu=0.1, lam=1.0, eta=0.5,
            promotion_test=promotion_test,
        )
        sgd_drqm = SGDLinDRQm(
            feat_dim=n_features, n_arms=n_arms,
            b_per_level=b_per_level, max_depth=d_cut,
            min_level=2, max_level=8, nu=0.1, lam=1.0, eta=0.5,
            seed=bandit_seed, promotion_test=promotion_test,
        )
        lints_drqm = LinTSDRQm(
            feat_dim=n_features, n_arms=n_arms,
            b_per_level=b_per_level, max_depth=d_cut,
            min_level=2, max_level=8, nu=0.1, lam=1.0, eta=0.5,
            seed=bandit_seed, promotion_test=promotion_test,
        )

        methods_order = ["TS", "LinTS", "SGD-LinTS",
                         "Counter-RQ", "SGD-RQ", "LinTS-RQ"]
        regret = {m: 0.0 for m in methods_order}
        wall_time = {m: 0.0 for m in methods_order}

        for t in range(n_rounds):
            idx = order[t]
            true_label = int(eval_labels[idx])
            x_flat = np.append(eval_features[idx], 1.0)
            tc = eval_codes[idx]
            res = residual_features[idx]

            # TS
            t0 = time.time()
            arm = int(ts.sample_means(ts_rng).argmax())
            reward = 1.0 if arm == true_label else 0.0
            ts.update(arm, reward)
            wall_time["TS"] += time.time() - t0
            regret["TS"] += 1.0 - reward

            # LinTS
            t0 = time.time()
            arm = lints.select_arm(x_flat)
            reward = 1.0 if arm == true_label else 0.0
            lints.update(x_flat, arm, reward)
            wall_time["LinTS"] += time.time() - t0
            regret["LinTS"] += 1.0 - reward

            # SGD-LinTS
            t0 = time.time()
            arm = sgd.select_arm(x_flat)
            reward = 1.0 if arm == true_label else 0.0
            sgd.update(x_flat, arm, reward)
            wall_time["SGD-LinTS"] += time.time() - t0
            regret["SGD-LinTS"] += 1.0 - reward

            # Counter-RQ
            t0 = time.time()
            arm = counter_drqm.select_arm(tc)
            reward = 1.0 if arm == true_label else 0.0
            counter_drqm.update(tc, arm, reward)
            wall_time["Counter-RQ"] += time.time() - t0
            regret["Counter-RQ"] += 1.0 - reward

            # SGD-RQ
            t0 = time.time()
            arm = sgd_drqm.select_arm(tc, res)
            reward = 1.0 if arm == true_label else 0.0
            sgd_drqm.update(tc, res, arm, reward)
            wall_time["SGD-RQ"] += time.time() - t0
            regret["SGD-RQ"] += 1.0 - reward

            # LinTS-RQ
            t0 = time.time()
            arm = lints_drqm.select_arm(tc, res)
            reward = 1.0 if arm == true_label else 0.0
            lints_drqm.update(tc, res, arm, reward)
            wall_time["LinTS-RQ"] += time.time() - t0
            regret["LinTS-RQ"] += 1.0 - reward

        for m in methods_order:
            all_regrets.setdefault(m, []).append(regret[m] / n_rounds)
            all_times.setdefault(m, []).append(
                wall_time[m] / n_rounds * 1e6
            )

        rates = {m: regret[m] / n_rounds * 100 for m in methods_order}
        print(
            f"  seed {run_idx+1}/{n_runs}: " +
            ", ".join(f"{m}={rates[m]:.1f}%" for m in methods_order),
            flush=True,
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"Regret rate (%) at T={n_rounds}, {n_runs} seeds:")
    print(f"{'Method':<16} {'Rate (%)':<16} {'us/step':<12} {'vs LinTS'}")
    print(f"{'-'*60}")

    lints_mean = np.mean(all_regrets["LinTS"]) * 100

    for m in methods_order:
        vals = np.array(all_regrets[m]) * 100
        times = np.array(all_times[m])
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        t_mean = np.mean(times)

        if m == "LinTS":
            gap = ""
        else:
            g = (mean_r - lints_mean) / lints_mean * 100
            gap = f"{g:+.1f}%"

        print(f"{m:<16} {mean_r:>5.1f} +/- {std_r:>4.1f}    {t_mean:>8.0f}    {gap}")

    # Promotions
    print(f"\nShadow promotions:")
    for name, obj in [("Counter-RQ", counter_drqm),
                      ("SGD-RQ", sgd_drqm),
                      ("LinTS-RQ", lints_drqm)]:
        log = obj.promotion_log
        if log:
            depths = [f"{old}->{new} at t={t}" for t, old, new in log]
            print(f"  {name}: {', '.join(depths)}")
        else:
            print(f"  {name}: stayed at depth {obj.active_depth}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E demo: contextual bandits on MiniBooNE"
    )
    parser.add_argument("--rq-n", type=int, default=5000,
                        help="Holdout for codebook (default: 5000)")
    parser.add_argument("--d-cut", type=int, default=4,
                        help="RQ depth (default: 4)")
    parser.add_argument("--nbits", type=int, default=4,
                        help="Bits per level, b=2^nbits (default: 4)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of seeds (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Cap eval rounds, 0=all (default: 0)")
    parser.add_argument("--promotion-test", type=str, default="ttest",
                        choices=["ttest", "freedman", "both"],
                        help="Shadow promotion test (default: ttest)")
    args = parser.parse_args()

    if args.promotion_test == "both":
        for pt in ["ttest", "freedman"]:
            print(f"\n{'='*70}")
            print(f"  PROMOTION TEST: {pt}")
            print(f"{'='*70}")
            run_demo(
                rq_n=args.rq_n,
                d_cut=args.d_cut,
                nbits=args.nbits,
                n_runs=args.n_runs,
                seed=args.seed,
                max_rounds=args.max_rounds,
                promotion_test=pt,
            )
    else:
        run_demo(
            rq_n=args.rq_n,
            d_cut=args.d_cut,
            nbits=args.nbits,
            n_runs=args.n_runs,
            seed=args.seed,
            max_rounds=args.max_rounds,
            promotion_test=args.promotion_test,
        )


if __name__ == "__main__":
    main()
