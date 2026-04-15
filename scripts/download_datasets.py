#!/usr/bin/env python3


"""Download tabular datasets and upload to Manifold for MAST consumption.

Downloads raw data files to the local cache, loads each dataset through
the standard loaders, serializes train/val/test splits, and uploads to
Manifold so MAST jobs (which lack local file access and internet) can
load them.

Usage:
    python3 scripts/download_datasets.py

    # Download only (skip Manifold upload):
    python3 scripts/download_datasets -- --no-upload.py

    # Upload only (skip download, assumes data already cached):
    python3 scripts/download_datasets -- --upload-only --datasets adult.py

    # Specific datasets:
    python3 scripts/download_datasets -- --datasets helena aloi letter.py

IMPORTANT: Run from a regular terminal, NOT through Claude Code,
because Claude Code's agent identity is blocked by the network filter
for external URLs (openml.org, archive.ics.uci.edu, etc.).
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import logging
from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from modules.data import (
    DATASET_REGISTRY,
    load_dataset,
    TabularDataset,
    TabularDatasetMetadata,
)

# Setup fwdproxy for Meta network BEFORE importing sklearn
os.environ["http_proxy"] = "http://fwdproxy:8080"
os.environ["https_proxy"] = "http://fwdproxy:8080"
os.environ["HTTP_PROXY"] = "http://fwdproxy:8080"
os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"

from sklearn.datasets import fetch_openml  # noqa: E402

logger = logging.getLogger(__name__)

# Cache directories
PKL_CACHE_DIR = Path(f"/data/users/{os.environ['USER']}/datasets")
PKL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_CACHE_DIR = Path(
    "./data/cache"
)
CSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MANIFOLD_DATA_ROOT: str = (
    "manifold://fi_platform_ml_infra_fluent2_bucket/tree/fi_trunk_tail/datasets"
)

_pathmgr: PathManager = PathManager()
_pathmgr.register_handler(ManifoldPathHandler())


def _download_openml_csv(
    name: str,
    openml_name: str | None = None,
    openml_id: int | None = None,
    target_column: str = "class",
    description: str = "",
) -> Path:
    """Download a dataset from OpenML and save as CSV."""
    output_path = CSV_CACHE_DIR / f"{name}.csv"
    if output_path.exists():
        print(f"{name} already exists at {output_path}")
        return output_path
    print(f"Downloading {name} ({description})...")
    if openml_id is not None:
        data = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    else:
        data = fetch_openml(name=openml_name, version=1, as_frame=True, parser="auto")
    df = data.frame
    n_classes = df[target_column].nunique() if target_column in df.columns else "?"
    print(f"  Shape: {df.shape}, Classes: {n_classes}")
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    return output_path


def _download_openml_pkl(
    name: str,
    openml_name: str | None = None,
    openml_id: int | None = None,
    description: str = "",
) -> Path:
    """Download a dataset from OpenML and save as pickle."""
    output_path = PKL_CACHE_DIR / f"{name}.pkl"
    if output_path.exists():
        print(f"{name} already exists at {output_path}")
        return output_path
    print(f"Downloading {name} ({description})...")
    if openml_id is not None:
        data = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    else:
        data = fetch_openml(name=openml_name, version=1, as_frame=True, parser="auto")
    with open(output_path, "wb") as f:
        pickle.dump({"data": data.data, "target": data.target, "frame": data.frame}, f)
    print(f"  Saved to: {output_path}")
    print(f"  Samples: {len(data.data)}")
    return output_path


# === Dataset downloaders ===


# Tier 1: Many classes (100+) — best for label efficiency
def download_aloi() -> Path:
    """ALOI: 1000 classes, ~108K samples, 128 features. OpenML 42396."""
    return _download_openml_csv(
        "aloi",
        openml_id=42396,
        target_column="class",
        description="1000 classes, ~108K samples",
    )


def download_helena() -> Path:
    """Helena: 100 classes, ~65K samples, 27 features. OpenML 41169."""
    return _download_openml_csv(
        "helena",
        openml_name="helena",
        target_column="class",
        description="100 classes, ~65K samples",
    )


def download_dionis() -> Path:
    """Dionis: 355 classes, ~416K samples, 60 features. OpenML 41167."""
    return _download_openml_csv(
        "dionis",
        openml_id=41167,
        target_column="class",
        description="355 classes, ~416K samples",
    )


# Tier 2: Moderate classes (10-26)
def download_letter() -> Path:
    """Letter Recognition: 26 classes, ~20K samples, 16 features. OpenML 6."""
    return _download_openml_csv(
        "letter",
        openml_id=6,
        target_column="lettr",
        description="26 classes, ~20K samples",
    )


def download_volkert() -> Path:
    """Volkert: 10 classes, ~58K samples, 180 features. OpenML 41166."""
    return _download_openml_csv(
        "volkert",
        openml_id=41166,
        target_column="class",
        description="10 classes, ~58K samples",
    )


def download_jannis() -> Path:
    """Jannis: 4 classes, ~83K samples, 54 features. OpenML 41168."""
    return _download_openml_csv(
        "jannis",
        openml_name="jannis",
        target_column="class",
        description="4 classes, ~83K samples",
    )


def download_pendigits() -> Path:
    """Pendigits: 10 classes, ~10,992 samples, 16 features. OpenML 32."""
    return _download_openml_csv(
        "pendigits",
        openml_id=32,
        target_column="class",
        description="10 classes, ~11K samples",
    )


def download_fashion_mnist() -> Path:
    """Fashion-MNIST: 10 classes, ~70,000 samples, 784 features. OpenML 40996."""
    return _download_openml_csv(
        "fashion_mnist",
        openml_id=40996,
        target_column="class",
        description="10 classes, ~70K samples",
    )


def download_texture() -> Path:
    """Texture: 11 classes, ~5,500 samples, 40 features. OpenML 40499."""
    return _download_openml_csv(
        "texture",
        openml_id=40499,
        target_column="class",
        description="11 classes, ~5.5K samples",
    )


def download_shuttle() -> Path:
    """Shuttle: 7 classes, ~58,000 samples, 9 features. OpenML 40685."""
    return _download_openml_csv(
        "shuttle",
        openml_id=40685,
        target_column="class",
        description="7 classes, ~58K samples",
    )


# Legacy datasets (pkl format)
def download_adult() -> Path:
    """Adult: 2 classes, ~48K samples. OpenML adult v2."""
    return _download_openml_pkl(
        "adult", openml_name="adult", description="2 classes, ~48K"
    )


def download_bank_marketing() -> Path:
    """Bank Marketing: 2 classes, ~45K samples. OpenML 1461."""
    return _download_openml_pkl(
        "bank_marketing", openml_id=1461, description="2 classes, ~45K"
    )


def download_german_credit() -> Path:
    """German Credit: 2 classes, ~1K samples. OpenML credit-g."""
    return _download_openml_pkl(
        "german_credit", openml_name="credit-g", description="2 classes, ~1K"
    )


# Registry of all downloaders
ALL_DOWNLOADERS: dict[str, Any] = {
    # Tier 1: many classes
    "aloi": download_aloi,
    "helena": download_helena,
    "dionis": download_dionis,
    # Tier 2: moderate classes
    "letter": download_letter,
    "volkert": download_volkert,
    "jannis": download_jannis,
    "pendigits": download_pendigits,
    "fashion_mnist": download_fashion_mnist,
    "texture": download_texture,
    "shuttle": download_shuttle,
    # Tier 3: few classes (legacy)
    "adult": download_adult,
    "bank_marketing": download_bank_marketing,
    "german_credit": download_german_credit,
}

# Recommended for NeurIPS paper
NEURIPS_DATASETS = [
    "helena",
    "aloi",
    "letter",
    "jannis",
    "volkert",
    "dionis",
    "pendigits",
    "fashion_mnist",
    "texture",
    "shuttle",
]


# =============================================================================
# Manifold upload
# =============================================================================


def _serialize_dataset(
    train_ds: TabularDataset,
    val_ds: TabularDataset,
    test_ds: TabularDataset,
    metadata: TabularDatasetMetadata,
) -> dict[str, object]:
    """Build the serialization dict for torch.save."""
    return {
        "train_continuous": train_ds.continuous,
        "train_categorical": train_ds.categorical,
        "train_labels": train_ds.labels,
        "val_continuous": val_ds.continuous,
        "val_categorical": val_ds.categorical,
        "val_labels": val_ds.labels,
        "test_continuous": test_ds.continuous,
        "test_categorical": test_ds.categorical,
        "test_labels": test_ds.labels,
        "metadata": dataclasses.asdict(metadata),
    }


def upload_dataset(name: str) -> bool:
    """Load a dataset locally, serialize, and upload to Manifold.

    Returns True on success, False on failure.
    """
    uri = f"{_MANIFOLD_DATA_ROOT}/{name}.pt"
    logger.info(f"Uploading {name} -> {uri}")

    try:
        train_ds, val_ds, test_ds, metadata = load_dataset(name)
    except Exception:
        logger.error(f"Failed to load dataset {name} locally", exc_info=True)
        return False

    data = _serialize_dataset(train_ds, val_ds, test_ds, metadata)

    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(data, tmp.name)
            tmp.flush()

            with open(tmp.name, "rb") as src, _pathmgr.open(uri, "wb") as dst:
                dst.write(src.read())

        if not _pathmgr.exists(uri):
            logger.error(f"Upload verification failed for {name}: not found at {uri}")
            return False

        logger.info(
            f"Uploaded {name} ({len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test)"
        )
        return True

    except Exception:
        logger.error(f"Failed to upload dataset {name}", exc_info=True)
        return False


def upload_all(datasets: list[str]) -> None:
    """Upload datasets to Manifold for MAST."""
    print(f"\nUploading {len(datasets)} datasets to Manifold...")

    # Ensure the Manifold directory exists (it won't auto-create on put).
    bucket = "fi_platform_ml_infra_fluent2_bucket"
    dir_path = "tree/fi_trunk_tail/datasets"
    try:
        subprocess.check_call(
            ["manifold", "mkdir", "-p", f"{bucket}/{dir_path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Ensured Manifold directory exists: {bucket}/{dir_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(
            f"manifold mkdir failed ({e}), uploads may fail if dir doesn't exist"
        )

    succeeded = 0
    failed = 0
    for name in datasets:
        if upload_dataset(name):
            succeeded += 1
        else:
            failed += 1
    print(
        f"\nUpload complete: {succeeded} succeeded, {failed} failed out of {len(datasets)}"
    )
    if failed > 0:
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

# Default dataset list: all registered datasets except synthetic
_DEFAULT_DATASETS: list[str] = [k for k in DATASET_REGISTRY if k != "synthetic"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download datasets and upload to Manifold for MAST.",
        epilog="See docs/datasets_for_neurips.md for dataset details.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=sorted(set(ALL_DOWNLOADERS) | set(DATASET_REGISTRY)) + ["all", "neurips"],
        help=(
            "Datasets to download. 'all' downloads everything. "
            "'neurips' downloads the recommended set for the paper. "
            "Default: all registered datasets except synthetic."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Download only, skip Manifold upload.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip download, only upload to Manifold (assumes data already cached).",
    )
    args = parser.parse_args()

    datasets: list[str] = (
        args.datasets if args.datasets is not None else _DEFAULT_DATASETS
    )
    if "all" in datasets:
        datasets = list(ALL_DOWNLOADERS.keys())
    elif "neurips" in datasets:
        datasets = NEURIPS_DATASETS

    if not args.upload_only:
        for name in datasets:
            if name in ALL_DOWNLOADERS:
                print(f"\n{'=' * 60}")
                ALL_DOWNLOADERS[name]()
        print(f"\n{'=' * 60}")
        print("Download complete!")

    if not args.no_upload:
        # Upload all datasets that are in DATASET_REGISTRY
        upload_datasets = [d for d in datasets if d in DATASET_REGISTRY]
        upload_all(upload_datasets)


if __name__ == "__main__":
    main()
