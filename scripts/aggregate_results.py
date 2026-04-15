#!/usr/bin/env python3


"""Aggregate results from parallel MAST sweep jobs.

Reads per-job JSON result files from Manifold (or local filesystem),
groups by dataset, computes mean/std per method, runs significance tests,
and writes aggregated output.

Usage:
    python3 scripts/aggregate_results.py -- \
        --manifold-prefix ./results/contrastive/ \
        --output-file results/contrastive/sweep_aggregated.json

    # Or from local files:
    python3 scripts/aggregate_results.py -- \
        --local-dir /tmp/fi_trunk_tail/contrastive/ \
        --output-file results/contrastive/sweep_aggregated.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import logging
from modules.significance import (
    compute_pairwise_significance,
    log_significance,
)

logger = logging.getLogger(__name__)

# All RQ-based methods (ours)
RQ_METHODS: frozenset[str] = frozenset(
    {
        "m-RQ",
        "m-RQ+Gaussian",
        "RQ-SupCon+Proto",
        "RQ-WeightedSupCon",
        "RQ-HardNegSupCon",
        "RQ-Proto+Weighted",
    }
)


@dataclass
class SingleRunResult:
    """Result from a single experiment run (matches run_all_experiments format)."""

    dataset: str
    seed: int
    method: str
    accuracy: float
    time_seconds: float
    codebook_time_seconds: float = 0.0
    d_cut: int | None = None
    reconstruction_error: float | None = None


def _parse_single_result(r: dict[str, Any]) -> SingleRunResult:
    """Parse a dict into a SingleRunResult."""
    return SingleRunResult(
        dataset=r["dataset"],
        seed=r["seed"],
        method=r["method"],
        accuracy=r["accuracy"],
        time_seconds=r["time_seconds"],
        codebook_time_seconds=r.get("codebook_time_seconds", 0.0),
        d_cut=r.get("d_cut"),
        reconstruction_error=r.get("reconstruction_error"),
    )


def _extract_results_from_json(obj: dict[str, Any]) -> list[SingleRunResult]:
    """Extract SingleRunResult entries from a parsed JSON object."""
    results: list[SingleRunResult] = []
    if "result" in obj:
        results.append(_parse_single_result(obj["result"]))
    elif "results" in obj and isinstance(obj["results"], list):
        for r in obj["results"]:
            if isinstance(r, dict) and "accuracy" in r:
                results.append(_parse_single_result(r))
    return results


def load_results_from_manifold(
    bucket: str,
    prefix: str,
) -> list[SingleRunResult]:
    """Load all result JSONs from a Manifold prefix (recursive)."""
    from io import BytesIO

    from manifold.clients.python import ManifoldClient

    results: list[SingleRunResult] = []

    with ManifoldClient.get_client(bucket) as client:
        dirs_to_visit = [prefix.rstrip("/")]
        visited: set[str] = set()

        while dirs_to_visit:
            current_dir = dirs_to_visit.pop()
            if current_dir in visited:
                continue
            visited.add(current_dir)

            try:
                for entry_name, _entry_meta in client.sync_ls(current_dir):
                    full_path = f"{current_dir}/{entry_name}"
                    if entry_name.endswith(".json"):
                        try:
                            output = BytesIO()
                            client.sync_get(full_path, output)
                            output.seek(0)
                            obj = json.loads(output.read())
                            results.extend(_extract_results_from_json(obj))
                        except Exception as e:
                            logger.warning(f"Failed to load {full_path}: {e}")
                    else:
                        dirs_to_visit.append(full_path)
            except Exception as e:
                logger.warning(f"Failed to list {current_dir}: {e}")

    logger.info(f"Loaded {len(results)} results from # manifold:// {bucket}/{prefix}")
    return results


def load_results_from_local(local_dir: str) -> list[SingleRunResult]:
    """Load all result JSONs from a local directory."""
    results: list[SingleRunResult] = []
    local_path = Path(local_dir)

    if not local_path.exists():
        logger.error(f"Local directory does not exist: {local_dir}")
        return results

    for json_file in local_path.rglob("*.json"):
        try:
            with open(json_file) as f:
                obj = json.load(f)
            results.extend(_extract_results_from_json(obj))
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    logger.info(f"Loaded {len(results)} results from {local_dir}")
    return results


def _compute_method_stats(
    all_results: list[SingleRunResult],
    dataset_name: str,
    method: str,
) -> dict[str, float] | None:
    """Compute summary statistics for a single (dataset, method) pair."""
    method_results = [
        r for r in all_results if r.dataset == dataset_name and r.method == method
    ]
    if not method_results:
        return None
    accuracies = [r.accuracy for r in method_results]
    times = [r.time_seconds for r in method_results]
    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "min": float(np.min(accuracies)),
        "max": float(np.max(accuracies)),
        "n_runs": len(accuracies),
        "mean_time_seconds": float(np.mean(times)),
    }


def _compute_dataset_significance(
    ds_results: list[SingleRunResult],
) -> dict[str, object] | None:
    """Compute pairwise significance tests for one dataset's results."""
    method_seed_acc: dict[str, dict[int, float]] = defaultdict(dict)
    for r in ds_results:
        method_seed_acc[r.method][r.seed] = r.accuracy

    all_seed_sets = [set(v.keys()) for v in method_seed_acc.values()]
    if not all_seed_sets:
        return None
    common_seeds = sorted(set.intersection(*all_seed_sets))
    if len(common_seeds) < 2:
        return None

    method_seed_values: dict[str, list[float]] = {
        method: [seed_acc[s] for s in common_seeds]
        for method, seed_acc in method_seed_acc.items()
    }

    return compute_pairwise_significance(
        method_seed_values,
        ours_methods=RQ_METHODS,
        higher_is_better=True,
    )


def aggregate_results(all_results: list[SingleRunResult]) -> dict[str, Any]:
    """Aggregate results: compute per-dataset, per-method summary statistics and significance tests."""

    datasets_sorted = sorted({r.dataset for r in all_results})
    methods_sorted = sorted({r.method for r in all_results})

    # Compute summary statistics
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_name in datasets_sorted:
        summary[dataset_name] = {}
        for method in methods_sorted:
            stats = _compute_method_stats(all_results, dataset_name, method)
            if stats:
                summary[dataset_name][method] = stats

    # Significance tests
    significance: dict[str, dict[str, object]] = {}
    for dataset_name in datasets_sorted:
        ds_results = [r for r in all_results if r.dataset == dataset_name]
        if ds_results:
            sig = _compute_dataset_significance(ds_results)
            if sig:
                significance[dataset_name] = sig

    # Build output
    output: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_total_results": len(all_results),
            "datasets": datasets_sorted,
            "methods": methods_sorted,
        },
        "results": [asdict(r) for r in all_results],
        "summary": summary,
    }

    if significance:
        output["significance"] = significance

    return output


def print_summary_table(output: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    summary = output["summary"]

    all_methods: set[str] = set()
    for ds_data in summary.values():
        all_methods.update(ds_data.keys())
    methods = sorted(all_methods)

    logger.info("\n" + "=" * 100)
    logger.info("AGGREGATED RESULTS: RQ-based (ours) vs Baselines (Accuracy %)")
    logger.info("=" * 100)

    header = f"{'Dataset':<20}"
    for m in methods:
        tag = "*" if m in RQ_METHODS else " "
        header += f"{m:>14}{tag}"
    logger.info(header)
    logger.info("-" * 100)

    for dataset, method_stats in summary.items():
        row = f"{dataset:<20}"
        for m in methods:
            stats = method_stats.get(m, {})
            if stats:
                mean = stats["mean"] * 100
                std = stats["std"] * 100
                n = int(stats["n_runs"])
                row += f" {mean:>6.2f}±{std:>4.2f}({n:>2d})"
            else:
                row += f"{'N/A':>15}"
        logger.info(row)

    logger.info("-" * 100)
    logger.info("* = ours (RQ-based)")

    # Log significance
    if "significance" in output:
        logger.info("\n" + "=" * 80)
        logger.info("SIGNIFICANCE TESTS")
        logger.info("=" * 80)
        for ds, sig in output["significance"].items():
            logger.info(f"\n  Dataset: {ds}")
            log_significance(sig, logger, metric_name="accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sweep results")
    parser.add_argument(
        "--manifold-prefix",
        type=str,
        default=None,
        help="Manifold path (# manifold:// bucket/prefix/)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory with result JSONs",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file for aggregated results",
    )
    args = parser.parse_args()

    if args.manifold_prefix is None and args.local_dir is None:
        parser.error("Must specify --manifold-prefix or --local-dir")

    # Load results
    all_results: list[SingleRunResult] = []

    if args.manifold_prefix is not None:
        # Parse # manifold:// bucket/prefix format
        prefix = args.manifold_prefix
        if prefix.startswith("# manifold:// "):
            prefix = prefix[len("# manifold:// ") :]
        parts = prefix.split("/", 1)
        bucket = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        all_results.extend(load_results_from_manifold(bucket, path))

    if args.local_dir is not None:
        all_results.extend(load_results_from_local(args.local_dir))

    if not all_results:
        logger.error("No results found!")
        return

    # Aggregate
    output = aggregate_results(all_results)

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    logger.info(f"\nAggregated results saved to: {output_path}")

    # Print summary
    print_summary_table(output)


if __name__ == "__main__":
    main()
