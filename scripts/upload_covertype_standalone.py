#!/usr/bin/env python3
"""One-off script: create covertype.pt and upload to Manifold.

Reads covtype.data locally, processes into train/val/test splits (same logic
as load_covertype in data/__init__.py), saves as .pt, uploads via manifold CLI.

No buck2 needed — uses only stdlib + pip packages available on devservers.

Usage:
    python3 scripts/upload_covertype_standalone.py
"""

import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COVTYPE_FILE = Path(
    "./data/cache/covtype.data"
)
MANIFOLD_BUCKET = "fi_platform_ml_infra_fluent2_bucket"
MANIFOLD_PATH = "tree/fi_trunk_tail/datasets/covertype.pt"


@dataclass
class TabularDatasetMetadata:
    name: str
    n_samples: int
    n_continuous: int
    n_categorical: int
    n_classes: int
    continuous_columns: list
    categorical_columns: list
    target_column: str
    class_names: list
    category_sizes: list


def main() -> None:
    if not COVTYPE_FILE.exists():
        print(f"ERROR: {COVTYPE_FILE} not found")
        sys.exit(1)

    print(f"Reading {COVTYPE_FILE} ({COVTYPE_FILE.stat().st_size / 1e6:.1f} MB)...")

    covtype_columns = (
        [
            "Elevation", "Aspect", "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        + [f"Wilderness_Area_{i}" for i in range(1, 5)]
        + [f"Soil_Type_{i}" for i in range(1, 41)]
        + ["Cover_Type"]
    )

    df = pd.read_csv(COVTYPE_FILE, header=None, names=covtype_columns)
    print(f"  Loaded: {len(df)} samples, {len(df.columns)} columns")

    continuous_columns = [
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    categorical_columns = (
        [f"Wilderness_Area_{i}" for i in range(1, 5)]
        + [f"Soil_Type_{i}" for i in range(1, 41)]
    )

    # Target: 1-7 -> 0-6
    y = df["Cover_Type"].values - 1
    class_names = [f"Type_{i}" for i in range(1, 8)]

    # Continuous features
    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Categorical features (already 0/1)
    categorical_data = df[categorical_columns].values.astype(np.int64)
    category_sizes = [2] * len(categorical_columns)

    # Split: train/val/test
    test_size, val_size, random_state = 0.2, 0.1, 42

    X_cont_train, X_cont_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(
            continuous_data, categorical_data, y,
            test_size=test_size, random_state=random_state, stratify=y,
        )
    )
    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = (
        train_test_split(
            X_cont_train, X_cat_train, y_train,
            test_size=val_size, random_state=random_state, stratify=y_train,
        )
    )

    print(f"  Split: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")

    metadata = TabularDatasetMetadata(
        name="covertype",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=len(categorical_columns),
        n_classes=7,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column="Cover_Type",
        class_names=class_names,
        category_sizes=category_sizes,
    )

    data = {
        "train_continuous": torch.tensor(continuous_data[: len(y_train)], dtype=torch.float32),
        "train_categorical": torch.tensor(X_cat_train, dtype=torch.long),
        "train_labels": torch.tensor(y_train, dtype=torch.long),
        "val_continuous": torch.tensor(X_cont_val, dtype=torch.float32),
        "val_categorical": torch.tensor(X_cat_val, dtype=torch.long),
        "val_labels": torch.tensor(y_val, dtype=torch.long),
        "test_continuous": torch.tensor(X_cont_test, dtype=torch.float32),
        "test_categorical": torch.tensor(X_cat_test, dtype=torch.long),
        "test_labels": torch.tensor(y_test, dtype=torch.long),
        "metadata": asdict(metadata),
    }

    # Fix: use the actual split data for train_continuous
    data["train_continuous"] = torch.tensor(X_cont_train, dtype=torch.float32)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(data, tmp.name)
        tmp_path = tmp.name

    size_mb = Path(tmp_path).stat().st_size / 1e6
    print(f"  Saved .pt: {tmp_path} ({size_mb:.1f} MB)")

    # Upload via manifold CLI
    print(f"  Uploading to manifold://{MANIFOLD_BUCKET}/{MANIFOLD_PATH}...")
    result = subprocess.run(
        ["manifold", "put", tmp_path, f"{MANIFOLD_BUCKET}/{MANIFOLD_PATH}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: manifold put failed: {result.stderr}")
        sys.exit(1)

    # Verify
    result = subprocess.run(
        ["manifold", "ls", f"{MANIFOLD_BUCKET}/{MANIFOLD_PATH}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and MANIFOLD_PATH.split("/")[-1] in result.stdout:
        print(f"  SUCCESS: covertype.pt uploaded and verified on Manifold")
    else:
        print(f"  WARNING: upload may have failed, verify manually:")
        print(f"    manifold ls {MANIFOLD_BUCKET}/{MANIFOLD_PATH}")

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
