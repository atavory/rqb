#!/usr/bin/env python3


"""Fiber purity by embedding type: how pure are RQ fibers at each depth?

For each dataset and embedding type (raw / lgbm / iforest):
1. Extract features via ``EmbeddingExtractor``.
2. Standardize with ``normalize_for_rq``.
3. Fit FAISS ``ResidualQuantizer(n_features, max_depth=8, nbits=8)``.
4. At each depth d=1..8 compute frequency-weighted mean purity:
   ``sum(max_c P(Y=c | Z_{1:d}) * |fiber|) / N``
5. Save per-dataset JSON to ``/tmp/fiber_purity/`` and optionally to Manifold.

Usage:
    python3 scripts/experiments/plot_fiber_purity.py -- \\
        --embedding-type lgbm --datasets adult
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
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

MANIFOLD_BUCKET = "fi_platform_ml_infra_fluent2_bucket"
MANIFOLD_RESULTS_PREFIX = "tree/fi_trunk_tail/results/fiber_purity"


def _write_json_to_manifold(data: dict[str, object], manifold_path: str) -> None:
    """Write a JSON dict to Manifold using the CLI."""
    full_path = f"{MANIFOLD_BUCKET}/{manifold_path}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["manifold", "put", "--overwrite", tmp_path, full_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"  Saved to # manifold:// (replace with local path) {full_path}")
    except Exception as e:
        print(f"  WARNING: Failed to write # manifold:// (replace with local path) {full_path}: {e}")
    finally:
        os.unlink(tmp_path)


def compute_fiber_purity(
    labels: np.ndarray, codes: np.ndarray
) -> float:
    """Frequency-weighted mean purity of RQ fibers.

    For each unique code (fiber) z, purity(z) = max_c P(Y=c | Z=z).
    Returns ``sum(purity(z) * |fiber_z|) / N``.
    """
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)
    code_keys = [tuple(c) for c in codes]
    n = len(labels)

    # Count (code, label) co-occurrences
    code_counts: dict[tuple[int, ...], int] = Counter(code_keys)  # pyre-ignore[6]
    joint_counts: dict[tuple[tuple[int, ...], int], int] = Counter(
        zip(code_keys, labels.tolist())
    )  # pyre-ignore[6]

    weighted_purity = 0.0
    for ck, cc in code_counts.items():
        # Find the most frequent label in this fiber
        max_count = 0
        for lab in set(labels.tolist()):
            jc = joint_counts.get((ck, lab), 0)
            if jc > max_count:
                max_count = jc
        weighted_purity += max_count  # max_count / cc * cc = max_count

    return weighted_purity / n


def main() -> None:
    import faiss  # pyre-ignore[21]

    parser = argparse.ArgumentParser(
        description="Fiber purity by embedding type across RQ depths"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="raw",
        choices=["raw", "lgbm", "iforest"],
        help="Embedding type: raw features, LightGBM leaves, or IsolationForest leaves",
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
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum RQ depth to evaluate (default: 8)",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=8,
        help="Bits per RQ level (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/fiber_purity",
        help="Local output directory for results",
    )
    parser.add_argument(
        "--no-manifold",
        action="store_true",
        help="Skip writing results to Manifold",
    )
    args = parser.parse_args()

    embedding_type: str = args.embedding_type
    datasets: list[str] = args.datasets
    max_depth: int = args.max_depth
    nbits: int = args.nbits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding type: {embedding_type}")
    print(f"Datasets: {datasets}")
    print(f"max_depth: {max_depth}, nbits: {nbits}")

    all_summary: list[dict[str, object]] = []

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
        print(
            f"  Loaded in {time.time() - t0:.1f}s: {n_samples} samples, "
            f"{n_features} features, {metadata.n_classes} classes"
        )

        # Fit RQ at max depth, then evaluate purity at each prefix depth
        print(f"  Fitting RQ (depth={max_depth}, nbits={nbits})...")
        t1 = time.time()
        rq = faiss.ResidualQuantizer(n_features, max_depth, nbits)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.train(features_np)
        codes = rq.compute_codes(features_np)
        print(f"  RQ fitted in {time.time() - t1:.1f}s")

        purities: dict[str, float] = {}
        for d in range(1, max_depth + 1):
            trunk_codes = codes[:, :d]
            purity = compute_fiber_purity(labels_np, trunk_codes)
            purities[str(d)] = round(purity, 6)
            print(f"    depth={d}: purity={purity:.4f}")

        ds_result: dict[str, object] = {
            "dataset": ds_name,
            "experiment": "fiber_purity",
            "embedding_type": embedding_type,
            "nbits": nbits,
            "max_depth": max_depth,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": metadata.n_classes,
            "purities": purities,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        all_summary.append(ds_result)

        # Save per-dataset JSON locally
        ds_dir = output_dir / embedding_type
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_path = ds_dir / f"{ds_name}.json"
        with open(ds_path, "w") as f:
            json.dump(ds_result, f, indent=2)
        print(f"  Saved to {ds_path}")

        # Write to Manifold
        if not args.no_manifold:
            _write_json_to_manifold(
                ds_result,
                f"{MANIFOLD_RESULTS_PREFIX}/{embedding_type}/{ds_name}.json",
            )

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Fiber purity by depth [embedding={embedding_type}]")
    print(f"{'=' * 80}")
    depths = list(range(1, max_depth + 1))
    hdr = f"{'Dataset':20s}" + "".join(f"  d={d}" for d in depths)
    print(hdr)
    print("-" * len(hdr))
    for entry in all_summary:
        ds = str(entry["dataset"])
        p = entry["purities"]
        parts = []
        for d in depths:
            val = p.get(str(d), 0.0) if isinstance(p, dict) else 0.0
            parts.append(f"  {val:.3f}" if isinstance(val, float) else f"  {val}")
        print(f"{ds:20s}{''.join(parts)}")

    # Save summary
    summary_path = output_dir / f"fiber_purity_{embedding_type}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    if not args.no_manifold:
        _write_json_to_manifold(
            {"results": all_summary},  # pyre-ignore[6]
            f"{MANIFOLD_RESULTS_PREFIX}/{embedding_type}/summary.json",
        )


if __name__ == "__main__":
    main()
