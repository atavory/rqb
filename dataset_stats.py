# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Compute basic statistics for all available datasets.

Loads each .pt dataset file, applies the same preprocessing as
collect_raw_features (standardize continuous, one-hot encode categorical),
and outputs a CSV with:
  dataset, d (effective feature dim after preprocessing), a (number of arms),
  n (total usable samples), d2a (d² × a), p_max (frequency of most common
  label — the best-arm-in-hindsight accuracy without features)
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch


DATASETS_DIR = os.path.expanduser(
    "~/fi_trunk_tail_results/devvm50798/datasets"
)

OUTPUT_CSV = os.path.expanduser(
    "~/fi_trunk_tail_results/dataset_stats.csv"
)

# Datasets used in the paper (skip variants like higgs_5m, higgs_full, etc.)
DATASETS = [
    "adult",
    "airlines_delay",
    "aloi",
    "bank_marketing",
    "bng_elevators",
    "bng_letter",
    "california_housing",
    "covertype",
    "dionis",
    "fashion_mnist",
    "german_credit",
    "helena",
    "hepmass",
    "higgs",
    "jannis",
    "kddcup99",
    "letter",
    "miniboone",
    "nyc_taxi",
    "pendigits",
    "poker_hand",
    "real_hepmass",
    "shuttle",
    "skin_segmentation",
    "susy",
    "synthetic_sigmoidal",
    "texture",
    "volkert",
    "year_prediction",
]


def load_stats(name: str) -> dict[str, object] | None:
    pt_path = os.path.join(DATASETS_DIR, f"{name}.pt")
    if not os.path.exists(pt_path):
        return None

    data = torch.load(pt_path, weights_only=False)

    # Collect continuous features across splits
    cont_parts = []
    for split in ("train", "val", "test"):
        key = f"{split}_continuous"
        if key in data and data[key] is not None:
            cont_parts.append(data[key])
    cont_np = torch.cat(cont_parts).numpy() if cont_parts else None

    # Collect categorical features across splits
    cat_parts = []
    for split in ("train", "val", "test"):
        key = f"{split}_categorical"
        if key in data and data[key] is not None and data[key].numel() > 0:
            cat_parts.append(data[key])
    cat_np = torch.cat(cat_parts).numpy() if cat_parts else None

    # Collect labels across splits
    y_parts = []
    for split in ("train", "val", "test"):
        key = f"{split}_labels"
        if key in data and data[key] is not None:
            y_parts.append(data[key])
    y = torch.cat(y_parts).numpy() if y_parts else None

    if y is None:
        return None

    # Compute effective d using same logic as collect_raw_features:
    # - continuous columns count as-is
    # - categorical columns with <=2 values: 1 column (binary)
    # - categorical columns with >2 values: n_categories columns (one-hot)
    d_cont = cont_np.shape[1] if cont_np is not None else 0
    d_cat_onehot = 0
    if cat_np is not None and cat_np.shape[1] > 0:
        for col_idx in range(cat_np.shape[1]):
            n_categories = int(cat_np[:, col_idx].max()) + 1
            if n_categories <= 2:
                d_cat_onehot += 1
            else:
                d_cat_onehot += n_categories

    d = d_cont + d_cat_onehot
    n = y.shape[0]
    a = len(np.unique(y))
    d2a = d * d * a

    # p_max: frequency of the most common label (best-arm-in-hindsight without features)
    _, counts = np.unique(y, return_counts=True)
    p_max = float(counts.max()) / n

    return {"dataset": name, "d": d, "a": a, "n": n, "d2a": d2a, "p_max": round(p_max, 4)}


def main() -> None:
    rows = []
    for name in DATASETS:
        stats = load_stats(name)
        if stats is None:
            print(f"  SKIP {name}: not found", file=sys.stderr)
            continue
        rows.append(stats)
        print(f"  {name}: d={stats['d']}, a={stats['a']}, n={stats['n']}, d2a={stats['d2a']}, p_max={stats['p_max']}")

    rows.sort(key=lambda r: r["d2a"])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "d", "a", "n", "d2a", "p_max"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} datasets to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
