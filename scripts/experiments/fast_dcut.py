#!/usr/bin/env python3


"""Fast d_cut diagnostic: features -> normalize -> FAISS RQ -> H(Y|Z).

No encoder, no bootstrap, no CV. Just the core question:
at which depth does H(Y|trunk_codes) stop decreasing?

Supports ``--embedding-type`` to compare raw features, LightGBM leaf
embeddings, and Isolation Forest leaf embeddings.

Usage:
    python3 scripts/experiments/fast_dcut.py
    python3 scripts/fast_dcut -- --embedding-type lgbm --datasets adult --nbits 4 8.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from modules.data import load_dataset
from modules.embeddings import (
    EmbeddingConfig,
    EmbeddingExtractor,
)
from modules.features import (
    collect_raw_features,
    normalize_for_rq,
)


def conditional_entropy(
    labels: np.ndarray, codes: np.ndarray, alpha: float = 1.0
) -> float:
    """H(Y|Z) with add-alpha smoothing."""
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)
    code_keys = [tuple(c) for c in codes]
    unique_labels = sorted(set(labels.tolist()))
    n_labels = len(unique_labels)

    code_counts: dict[tuple[int, ...], int] = Counter(code_keys)  # pyre-ignore[6]
    joint_counts: dict[tuple[tuple[int, ...], int], int] = Counter(
        zip(code_keys, labels.tolist())
    )  # pyre-ignore[6]

    h = 0.0
    n = len(labels)
    for ck, cc in code_counts.items():
        p_z = cc / n
        for lab in unique_labels:
            jc = joint_counts.get((ck, lab), 0)
            p_y_given_z = (jc + alpha) / (cc + alpha * n_labels)
            if p_y_given_z > 0:
                h -= p_z * p_y_given_z * math.log(p_y_given_z)
    return h


def marginal_entropy(labels: np.ndarray, alpha: float = 1.0) -> float:
    """H(Y)."""
    counts = Counter(labels.tolist())
    unique = sorted(counts.keys())
    n = len(labels)
    n_labels = len(unique)
    h = 0.0
    for lab in unique:
        p = (counts[lab] + alpha) / (n + alpha * n_labels)
        if p > 0:
            h -= p * math.log(p)
    return h


def main() -> None:
    import faiss  # pyre-ignore[21]

    parser = argparse.ArgumentParser(
        description="Fast d_cut diagnostic across embedding types"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="raw",
        choices=["raw", "lgbm", "iforest"],
        help="Embedding type: raw features, LightGBM leaves, or IsolationForest leaves",
    )
    parser.add_argument(
        "--nbits",
        nargs="+",
        type=int,
        default=[2, 3, 4, 6, 8],
        help="Bits-per-level values to sweep (default: 2 3 4 6 8)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "adult",
            "bank_marketing",
            "covertype",
            "higgs",
            "helena",
            "jannis",
            "volkert",
            "aloi",
            "letter",
            "dionis",
        ],
        help="Datasets to evaluate",
    )
    args = parser.parse_args()

    embedding_type: str = args.embedding_type
    nbits_list: list[int] = args.nbits
    datasets: list[str] = args.datasets
    max_depth = 6
    output_dir = Path("/tmp/dcut_fast")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding type: {embedding_type}")
    print(f"nbits values: {nbits_list}")
    print(f"Datasets: {datasets}")

    all_results: dict[str, dict[str, object]] = {}

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}  (embedding_type={embedding_type})")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            train_dataset, _, _, metadata = load_dataset(ds_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        # Extract raw features
        features_np, labels_np = collect_raw_features(train_dataset)

        # Apply embedding extraction if not raw
        if embedding_type != "raw":
            print(f"  Fitting {embedding_type} embeddings...")
            extractor = EmbeddingExtractor(
                EmbeddingConfig(embedding_type=embedding_type)
            )
            y_train = labels_np if embedding_type == "lgbm" else None
            features_np = extractor.fit_transform(features_np, y_train=y_train)
            print(f"  Embedding shape: {features_np.shape}")

        # Normalize before RQ
        features_np, _, _ = normalize_for_rq(features_np)

        n_samples, n_features = features_np.shape
        h_y = marginal_entropy(labels_np)
        print(
            f"  Loaded in {time.time() - t0:.1f}s: {n_samples} samples, "
            f"{n_features} features, {metadata.n_classes} classes, H(Y)={h_y:.4f}"
        )

        ds_results: dict[str, object] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": metadata.n_classes,
            "h_y": h_y,
            "embedding_type": embedding_type,
        }
        nbits_results: dict[int, dict[str, object]] = {}

        for nbits in nbits_list:
            K = 2**nbits
            print(f"\n  nbits={nbits} (K={K}):")
            depth_data: list[dict[str, object]] = []
            recommended = 1

            for d in range(1, max_depth + 1):
                n_ctx_max = K**d
                if n_ctx_max > n_samples * 10:
                    print(f"    d={d}: SKIP (K^d={n_ctx_max:,} >> N={n_samples:,})")
                    break

                t1 = time.time()
                rq = faiss.ResidualQuantizer(n_features, d, nbits)
                rq.train_type = faiss.ResidualQuantizer.Train_default
                rq.train(features_np)
                codes = rq.compute_codes(features_np)
                trunk_codes = codes[:, :d]

                h_yz = conditional_entropy(labels_np, trunk_codes)
                n_unique = len(set(tuple(c) for c in trunk_codes))
                spc = n_samples / max(n_unique, 1)
                ig = h_y - h_yz
                nmi = ig / h_y if h_y > 0 else 0.0
                elapsed = time.time() - t1

                print(
                    f"    d={d}: H(Y|Z)={h_yz:.4f}  IG={ig:.4f}  NMI={nmi:.4f}  "
                    f"ctxs={n_unique:,}  samp/ctx={spc:.1f}  [{elapsed:.1f}s]"
                )
                depth_data.append(
                    {
                        "depth": d,
                        "h_yz": float(h_yz),
                        "info_gain": float(ig),
                        "nmi": float(nmi),
                        "n_unique": n_unique,
                        "samples_per_ctx": float(spc),
                    }
                )

            # Recommend d_cut: elbow in marginal info gain
            if len(depth_data) >= 2:
                mg = []
                for i, dd in enumerate(depth_data):
                    if i == 0:
                        mg.append(dd["info_gain"])
                    else:
                        mg.append(float(depth_data[i - 1]["h_yz"]) - float(dd["h_yz"]))
                thresh = 0.1 * mg[0] if mg[0] > 0 else 0.01
                recommended = depth_data[-1]["depth"]
                for i, m in enumerate(mg):
                    if i > 0 and m < thresh:
                        recommended = depth_data[i - 1]["depth"]
                        break
                # Feasibility: 50K budget, 10 obs/ctx
                for dd in depth_data:
                    n_unique = int(dd["n_unique"])
                    if 50000 / max(n_unique, 1) < 10:
                        idx = depth_data.index(dd)
                        fmax = int(depth_data[max(0, idx - 1)]["depth"])
                        recommended = min(int(recommended), fmax)
                        break
            elif depth_data:
                recommended = depth_data[0]["depth"]

            print(f"  --> d_cut = {recommended}")
            nbits_results[nbits] = {
                "recommended_dcut": recommended,
                "depths": depth_data,
            }

        ds_results["nbits_results"] = nbits_results
        all_results[ds_name] = ds_results

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Recommended d_cut per (dataset, nbits) [embedding={embedding_type}]")
    print(f"{'=' * 80}")
    hdr = f"{'Dataset':20s}" + "".join(f"  nb={nb}" for nb in nbits_list)
    print(hdr)
    print("-" * len(hdr))
    for ds in datasets:
        if ds not in all_results:
            print(f"{ds:20s}  FAILED")
            continue
        r = all_results[ds]
        parts = []
        for nb in nbits_list:
            nbr_obj = r.get("nbits_results", {}) if isinstance(r, dict) else {}
            nbr = nbr_obj if isinstance(nbr_obj, dict) else {}
            nb_obj = nbr.get(nb, {})
            nb_data = nb_obj if isinstance(nb_obj, dict) else {}
            dc = nb_data.get("recommended_dcut", "?")
            parts.append(f"  d={dc:>3}")
        print(f"{ds:20s}{''.join(parts)}")

    out_path = output_dir / f"fast_dcut_{embedding_type}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
