#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Minimal demo: trunk-tail contextual bandits on synthetic data.

Generates a synthetic Gaussian-cluster dataset, trains an RQ codebook,
encodes contexts, and runs a bandit loop comparing 6 methods.

Usage:
    buck run @mode/opt mitra/projects/fi_trunk_tail/scripts/experiments/algs:run_demo
"""

from __future__ import annotations

import argparse

import numpy as np
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.codebook import (
    compute_residual_features,
    encode,
    train_rq_codebook,
)
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.lints import LinTSBaseline
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.nig_stats import NIGStats
from mitra.projects.fi_trunk_tail.scripts.experiments.algs.sgd_lints import SGDLinTS
from mitra.projects.fi_trunk_tail.scripts.experiments.drq_bandits import (
    CounterDRQm,
    LinTSDRQm,
    SGDLinDRQm,
)


def _generate_synthetic(
    n_samples: int = 20000,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian-cluster data.

    Returns (features, labels) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, n_features) * 3
    features_list = []
    labels_list = []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        features_list.append(rng.randn(per_class, n_features).astype(np.float32) + centers[c])
        labels_list.extend([c] * per_class)
    features = np.vstack(features_list)
    labels = np.array(labels_list, dtype=np.int64)
    perm = rng.permutation(len(features))
    return features[perm], labels[perm]


def _add_intercept(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)


def run_demo(
    n_samples: int = 20000,
    n_features: int = 10,
    n_classes: int = 5,
    rq_n: int = 5000,
    d_cut: int = 4,
    nbits: int = 4,
    n_runs: int = 3,
    seed: int = 42,
) -> None:
    # --- Generate synthetic data ---
    features, labels = _generate_synthetic(n_samples, n_features, n_classes, seed)
    n_total = len(features)
    n_arms = n_classes
    b_per_level = 2 ** nbits
    print(f"Synthetic data: N={n_total}, d={n_features}, C={n_arms}")

    # --- Inductive split: holdout for codebook, rest for bandit ---
    rng_split = np.random.RandomState(42)
    perm = rng_split.permutation(n_total)
    codebook_idx = perm[:rq_n]
    eval_idx = perm[rq_n:]
    n_rounds = len(eval_idx)
    print(f"Codebook: {rq_n}, Eval: {n_rounds}, d_cut={d_cut}, nbits={nbits}")

    # --- Train RQ codebook on holdout ---
    rq, centroids = train_rq_codebook(features[codebook_idx], d_cut, nbits)

    # --- Encode eval set ---
    eval_features = features[eval_idx]
    eval_labels = labels[eval_idx]
    eval_codes = encode(rq, eval_features, d_cut, nbits)
    residual_features = compute_residual_features(eval_features, eval_codes, centroids, d_cut)

    feature_dim = eval_features.shape[1] + 1  # +1 for intercept

    # --- Run bandit seeds ---
    all_regrets: dict[str, list[float]] = {}

    for run_idx in range(n_runs):
        bandit_seed = seed + run_idx
        rng = np.random.RandomState(bandit_seed)
        order = rng.permutation(n_rounds)

        ts = NIGStats(n_arms=n_arms)
        lints = LinTSBaseline(
            input_dim=feature_dim, n_arms=n_arms,
            lambda_prior=1.0, nu=0.1, rng=np.random.RandomState(bandit_seed),
        )
        sgd = SGDLinTS(
            input_dim=feature_dim, n_arms=n_arms,
            nu=0.1, lam=1.0, seed=bandit_seed,
        )
        counter_drqm = CounterDRQm(
            n_arms=n_arms, b_per_level=b_per_level, max_depth=d_cut,
            min_level=2, max_level=8, nu=0.1, lam=1.0, eta=0.5,
        )
        sgd_drqm = SGDLinDRQm(
            feat_dim=n_features, n_arms=n_arms,
            b_per_level=b_per_level, max_depth=d_cut,
            min_level=2, max_level=8, nu=0.1, lam=1.0, eta=0.5,
            seed=bandit_seed,
        )
        lints_drqm = LinTSDRQm(
            feat_dim=n_features, n_arms=n_arms,
            b_per_level=b_per_level, max_depth=d_cut,
            min_level=2, max_level=8, nu=0.1, lam=1.0, eta=0.5,
            seed=bandit_seed,
        )

        methods = {
            "TS": ts,
            "LinTS": lints,
            "SGD-LinTS": sgd,
            "Counter-DRQm": counter_drqm,
            "SGD-LinTS-DRQm": sgd_drqm,
            "LinTS-DRQm": lints_drqm,
        }
        regret = {m: 0.0 for m in methods}

        for t in range(n_rounds):
            idx = order[t]
            true_label = int(eval_labels[idx])
            emb = _add_intercept(eval_features[idx:idx+1])
            tc = eval_codes[idx]
            res = residual_features[idx]

            arm = ts.sample_means().argmax().item()
            reward = 1.0 if arm == true_label else 0.0
            ts.update(arm, reward)
            regret["TS"] += 1.0 - reward

            arm = lints.select_arm(emb)
            reward = 1.0 if arm == true_label else 0.0
            lints.update(emb, arm, reward)
            regret["LinTS"] += 1.0 - reward

            arm = sgd.select_arm(emb)
            reward = 1.0 if arm == true_label else 0.0
            sgd.update(emb, arm, reward)
            regret["SGD-LinTS"] += 1.0 - reward

            arm = counter_drqm.select_arm(tc)
            reward = 1.0 if arm == true_label else 0.0
            counter_drqm.update(tc, arm, reward)
            regret["Counter-DRQm"] += 1.0 - reward

            arm = sgd_drqm.select_arm(tc, res)
            reward = 1.0 if arm == true_label else 0.0
            sgd_drqm.update(tc, res, arm, reward)
            regret["SGD-LinTS-DRQm"] += 1.0 - reward

            arm = lints_drqm.select_arm(tc, res)
            reward = 1.0 if arm == true_label else 0.0
            lints_drqm.update(tc, res, arm, reward)
            regret["LinTS-DRQm"] += 1.0 - reward

        for m, r in regret.items():
            all_regrets.setdefault(m, []).append(r)

        print(f"  seed {run_idx+1}/{n_runs}: " +
              ", ".join(f"{m}={regret[m]:.0f}" for m in methods), flush=True)

    # --- Summary ---
    print(f"\nFinal regret at T={n_rounds} ({n_runs} seeds):")
    for m in methods:
        vals = all_regrets[m]
        print(f"  {m:<16} {np.mean(vals):>10.1f} ± {np.std(vals):>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trunk-tail bandit demo (synthetic data)")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--rq-n", type=int, default=5000)
    parser.add_argument("--d-cut", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_demo(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        rq_n=args.rq_n,
        d_cut=args.d_cut,
        nbits=args.nbits,
        n_runs=args.n_runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
