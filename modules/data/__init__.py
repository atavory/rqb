

"""Data loading utilities for tabular experiments.

This module provides data loaders for tabular datasets used in experiments:
    - Adult (UCI Adult Income): Binary classification, ~48K samples
    - Covertype: Multi-class classification, ~581K samples
    - Synthetic: Generated data for quick testing

Example:
    >>> from modules.data.tabular import load_adult
    >>> train_data, test_data, metadata = load_adult()
    >>> print(f"Features: {metadata['n_continuous']} continuous, {metadata['n_categorical']} categorical")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import logging
from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _setup_fwdproxy() -> None:
    """Configure fwdproxy for external internet access from Meta devservers.

    This sets environment variables that urllib (used by sklearn's fetch_openml)
    reads to route HTTP(S) traffic through Meta's forward proxy.
    """
    os.environ["http_proxy"] = "http://fwdproxy:8080"
    os.environ["https_proxy"] = "http://fwdproxy:8080"
    os.environ["HTTP_PROXY"] = "http://fwdproxy:8080"
    os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"
    os.environ["no_proxy"] = (
    )


# =============================================================================
# Manifold-based dataset cache
# =============================================================================

_MANIFOLD_DATA_ROOT: str = (
    "./data/cache"
)

_MANIFOLD_RAW_CACHE: str = f"{_MANIFOLD_DATA_ROOT}/raw_cache"

_pathmgr: PathManager = PathManager()
_pathmgr.register_handler(ManifoldPathHandler())


def _fetch_raw_from_manifold(filename: str, local_path: Path) -> bool:
    """Download a raw data file from Manifold raw_cache to a local path.

    Returns True if the file was successfully fetched, False otherwise.
    """
    uri = f"{_MANIFOLD_RAW_CACHE}/{filename}"
    try:
        if not _pathmgr.exists(uri):
            logger.debug(f"Raw file {filename} not found on Manifold at {uri}")
            return False
        logger.info(f"Fetching raw data from Manifold: {uri} -> {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with _pathmgr.open(uri, "rb") as src, open(local_path, "wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        logger.info(f"Fetched {filename} ({local_path.stat().st_size:,} bytes)")
        return True
    except Exception:
        logger.warning(f"Failed to fetch {filename} from Manifold", exc_info=True)
        return False


def _load_from_manifold(
    name: str,
) -> (
    tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata] | None
):
    """Try to load a pre-processed dataset from Manifold.

    Returns the (train, val, test, metadata) tuple if the dataset exists
    on Manifold, or None if it cannot be loaded (missing, network error, etc.).
    """
    uri = f"{_MANIFOLD_DATA_ROOT}/{name}.pt"
    try:
        if not _pathmgr.exists(uri):
            logger.debug(f"Dataset {name} not found on Manifold at {uri}")
            return None

        logger.info(f"Loading dataset {name} from Manifold: {uri}")
        with _pathmgr.open(uri, "rb") as f:
            data = torch.load(f, weights_only=False)

        def _to_numpy(t: torch.Tensor | None) -> np.ndarray | None:
            return t.numpy() if t is not None else None

        train_ds = TabularDataset(
            _to_numpy(data["train_continuous"]),
            _to_numpy(data["train_categorical"]),
            data["train_labels"].numpy(),
        )
        val_ds = TabularDataset(
            _to_numpy(data["val_continuous"]),
            _to_numpy(data["val_categorical"]),
            data["val_labels"].numpy(),
        )
        test_ds = TabularDataset(
            _to_numpy(data["test_continuous"]),
            _to_numpy(data["test_categorical"]),
            data["test_labels"].numpy(),
        )

        metadata = TabularDatasetMetadata(**data["metadata"])

        logger.info(
            f"Loaded {name} from Manifold: {metadata.n_samples} samples, "
            f"{metadata.n_continuous} continuous, {metadata.n_categorical} categorical"
        )
        return train_ds, val_ds, test_ds, metadata

    except Exception:
        logger.warning(
            f"Failed to load dataset {name} from Manifold, falling back to local loader",
            exc_info=True,
        )
        return None


@dataclass
class TabularDatasetMetadata:
    """Metadata for a tabular dataset."""

    name: str
    n_samples: int
    n_continuous: int
    n_categorical: int
    n_classes: int
    continuous_columns: list[str]
    categorical_columns: list[str]
    target_column: str
    class_names: list[str]
    category_sizes: list[int]  # Number of categories per categorical feature


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data.

    Supports mixed continuous and categorical features.

    Attributes:
        continuous: Continuous features, shape (n_samples, n_continuous).
        categorical: Categorical features, shape (n_samples, n_categorical).
        labels: Target labels, shape (n_samples,).
    """

    continuous: torch.Tensor | None
    categorical: torch.Tensor | None
    labels: torch.Tensor

    def __init__(
        self,
        continuous: np.ndarray | None,  # pyre-ignore[11]
        categorical: np.ndarray | None,  # pyre-ignore[11]
        labels: np.ndarray,  # pyre-ignore[11]
    ) -> None:
        """Initialize dataset.

        Args:
            continuous: Continuous features or None if no continuous features.
            categorical: Categorical features (integer encoded) or None.
            labels: Target labels.
        """
        self.continuous = (
            torch.tensor(continuous, dtype=torch.float32)
            if continuous is not None
            else None
        )
        self.categorical = (
            torch.tensor(categorical, dtype=torch.long)
            if categorical is not None
            else None
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of (features_dict, label) where features_dict contains
            'continuous' and/or 'categorical' keys.
        """
        features: dict[str, torch.Tensor] = {}
        if self.continuous is not None:
            features["cont_features"] = self.continuous[idx]
        if self.categorical is not None:
            features["cat_features"] = self.categorical[idx]

        return features, self.labels[idx]


def load_adult(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the UCI Adult Income dataset.

    The Adult dataset predicts whether income exceeds $50K/year based on
    census data. Features include age, workclass, education, occupation, etc.

    Args:
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.
        data_dir: Optional directory to cache downloaded data.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        raise ImportError(
            "sklearn is required for loading Adult dataset. "
            "Install with: pip install scikit-learn"
        ) from e

    # Configure fwdproxy for external internet access from Meta devservers
    _setup_fwdproxy()

    # Fetch from OpenML
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame

    # The target column in OpenML adult v2 is 'class' not 'income'
    # Detect the target column name
    target_column = "class" if "class" in df.columns else "income"
    if target_column not in df.columns:
        # Fallback: use the last column or the target from the bunch
        if hasattr(adult, "target") and adult.target is not None:
            df["target"] = adult.target
            target_column = "target"
        else:
            raise ValueError(
                f"Could not find target column. Available columns: {list(df.columns)}"
            )
    continuous_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Handle missing values
    df = df.dropna()

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_column].values)
    class_names = list(target_encoder.classes_)

    # Process continuous features
    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Process categorical features
    categorical_encoders: list[LabelEncoder] = []
    categorical_data: list[np.ndarray] = []
    category_sizes: list[int] = []

    for col in categorical_columns:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(df[col].values)
        categorical_data.append(encoded)
        categorical_encoders.append(encoder)
        category_sizes.append(len(encoder.classes_))

    categorical_data = np.column_stack(categorical_data).astype(np.int64)

    # Train/val/test split
    X_cont_train, X_cont_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(
            continuous_data,
            categorical_data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_cont_train,
        X_cat_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Create datasets
    train_dataset = TabularDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = TabularDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = TabularDataset(X_cont_test, X_cat_test, y_test)

    metadata = TabularDatasetMetadata(
        name="adult",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=len(categorical_columns),
        n_classes=len(class_names),
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=category_sizes,
    )

    return train_dataset, val_dataset, test_dataset, metadata


def load_covertype(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Covertype dataset.

    The Covertype dataset predicts forest cover type from cartographic
    variables. 7 classes, ~581K samples.

    Args:
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.
        max_samples: Optional limit on number of samples (for quick testing).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    import pandas as pd

    # First try local cache (downloaded from UCI)
    cache_dir = Path(
        "./data/cache"
    )
    local_file = cache_dir / "covtype.data"

    # UCI Covertype column names
    covtype_columns = (
        [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        + [f"Wilderness_Area_{i}" for i in range(1, 5)]
        + [f"Soil_Type_{i}" for i in range(1, 41)]
        + ["Cover_Type"]
    )

    if local_file.exists():
        df = pd.read_csv(local_file, header=None, names=covtype_columns)
    else:
        # Try Manifold raw cache (for MAST where local files don't exist)
        tmp_file = Path("/tmp/raw_cache/covtype.data")
        if not tmp_file.exists() and _fetch_raw_from_manifold("covtype.data", tmp_file):
            df = pd.read_csv(tmp_file, header=None, names=covtype_columns)
        elif tmp_file.exists():
            df = pd.read_csv(tmp_file, header=None, names=covtype_columns)
        else:
            # Fall back to sklearn fetch
            try:
                from sklearn.datasets import fetch_covtype
            except ImportError as e:
                raise ImportError(
                    "sklearn is required for loading Covertype dataset. "
                    "Install with: pip install scikit-learn"
                ) from e

            data = fetch_covtype(as_frame=True, data_home="/tmp/scikit_learn_data")
            df = data.frame

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    target_column = "Cover_Type"

    # 10 truly continuous cartographic features
    continuous_columns = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    # 44 binary indicator features (4 Wilderness Area + 40 Soil Type)
    categorical_columns = [f"Wilderness_Area_{i}" for i in range(1, 5)] + [
        f"Soil_Type_{i}" for i in range(1, 41)
    ]

    # Target is already numeric 1-7, convert to 0-6
    y = df[target_column].values - 1
    class_names = [f"Type_{i}" for i in range(1, 8)]

    # Process continuous features
    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Categorical features are already 0/1 integers
    categorical_data = df[categorical_columns].values.astype(np.int64)
    category_sizes = [2] * len(categorical_columns)

    # Train/val/test split
    X_cont_train, X_cont_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(
            continuous_data,
            categorical_data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_cont_train,
        X_cat_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    train_dataset = TabularDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = TabularDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = TabularDataset(X_cont_test, X_cat_test, y_test)

    metadata = TabularDatasetMetadata(
        name="covertype",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=len(categorical_columns),
        n_classes=7,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=category_sizes,
    )

    return train_dataset, val_dataset, test_dataset, metadata


def load_synthetic(
    n_samples: int = 10000,
    n_continuous: int = 10,
    n_categorical: int = 0,
    n_classes: int = 5,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Generate synthetic tabular data for testing.

    Creates Gaussian clusters with optional categorical features.

    Args:
        n_samples: Total number of samples.
        n_continuous: Number of continuous features.
        n_categorical: Number of categorical features.
        n_classes: Number of classes.
        random_state: Random seed.
        test_size: Fraction for test set.
        val_size: Fraction of training data for validation.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    np.random.seed(random_state)

    # Generate class centers
    centers = np.random.randn(n_classes, n_continuous) * 3

    # Generate samples
    X_cont: list[np.ndarray] = []
    y: list[int] = []
    for i in range(n_classes):
        n = n_samples // n_classes
        X_cont.append(np.random.randn(n, n_continuous) + centers[i])
        y.extend([i] * n)

    X_cont_arr = np.vstack(X_cont).astype(np.float32)
    y_arr = np.array(y)

    # Shuffle
    perm = np.random.permutation(len(X_cont_arr))
    X_cont_arr = X_cont_arr[perm]
    y_arr = y_arr[perm]

    # Generate categorical features if requested
    X_cat = None
    category_sizes_synth: list[int] = []
    if n_categorical > 0:
        X_cat_list: list[np.ndarray] = []
        for _ in range(n_categorical):
            n_cats = np.random.randint(3, 10)
            category_sizes_synth.append(n_cats)
            X_cat_list.append(np.random.randint(0, n_cats, len(X_cont_arr)))
        X_cat = np.column_stack(X_cat_list).astype(np.int64)

    # Split
    if X_cat is not None:
        X_cont_train, X_cont_test, X_cat_train, X_cat_test, y_train, y_test = (
            train_test_split(
                X_cont_arr, X_cat, y_arr, test_size=test_size, random_state=random_state
            )
        )
        X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = (
            train_test_split(
                X_cont_train,
                X_cat_train,
                y_train,
                test_size=val_size,
                random_state=random_state,
            )
        )
    else:
        X_cont_train, X_cont_test, y_train, y_test = train_test_split(
            X_cont_arr, y_arr, test_size=test_size, random_state=random_state
        )
        X_cont_train, X_cont_val, y_train, y_val = train_test_split(
            X_cont_train, y_train, test_size=val_size, random_state=random_state
        )
        X_cat_train = X_cat_val = X_cat_test = None

    train_dataset = TabularDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = TabularDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = TabularDataset(X_cont_test, X_cat_test, y_test)

    metadata = TabularDatasetMetadata(
        name="synthetic",
        n_samples=n_samples,
        n_continuous=n_continuous,
        n_categorical=n_categorical,
        n_classes=n_classes,
        continuous_columns=[f"cont_{i}" for i in range(n_continuous)],
        categorical_columns=[f"cat_{i}" for i in range(n_categorical)],
        target_column="target",
        class_names=[f"class_{i}" for i in range(n_classes)],
        category_sizes=category_sizes_synth,
    )

    return train_dataset, val_dataset, test_dataset, metadata


def load_bank_marketing(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the UCI Bank Marketing dataset.

    The Bank Marketing dataset predicts whether a client will subscribe
    to a term deposit (Y=1) based on marketing campaign data.

    Trunk features (stable): job, marital, education, default, housing, loan
    Tail features (variance): duration, campaign, pdays, previous, euribor3m

    Args:
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    import pandas as pd

    # First try local cache (downloaded from UCI)
    # Use relative path
    cache_dir = Path(
        "./data/cache"
    )
    local_file = cache_dir / "bank-full.csv"

    if local_file.exists():
        df = pd.read_csv(local_file, sep=";")
        target_column = "y"
    else:
        # Fall back to OpenML
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as e:
            raise ImportError(
                "sklearn is required for loading Bank Marketing dataset. "
                "Install with: pip install scikit-learn"
            ) from e

        _setup_fwdproxy()

        # Fetch from OpenML (Bank Marketing dataset, ID 1461)
        bank = fetch_openml(data_id=1461, as_frame=True, parser="auto")
        df = bank.frame
        target_column = "y" if "y" in df.columns else "Class"

    # Define feature groups (trunk = stable, tail = variance)
    continuous_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    categorical_columns = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    # Filter to available columns
    continuous_columns = [c for c in continuous_columns if c in df.columns]
    categorical_columns = [c for c in categorical_columns if c in df.columns]

    # Handle missing values
    df = df.dropna()

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_column].values)
    class_names = list(target_encoder.classes_)

    # Process continuous features
    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Process categorical features
    categorical_encoders: list[LabelEncoder] = []
    categorical_data_list: list[np.ndarray] = []
    category_sizes: list[int] = []

    for col in categorical_columns:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(df[col].values.astype(str))
        categorical_data_list.append(encoded)
        categorical_encoders.append(encoder)
        category_sizes.append(len(encoder.classes_))

    categorical_data = np.column_stack(categorical_data_list).astype(np.int64)

    # Train/val/test split
    X_cont_train, X_cont_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(
            continuous_data,
            categorical_data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_cont_train,
        X_cat_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Create datasets
    train_dataset = TabularDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = TabularDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = TabularDataset(X_cont_test, X_cat_test, y_test)

    metadata = TabularDatasetMetadata(
        name="bank_marketing",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=len(categorical_columns),
        n_classes=len(class_names),
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=category_sizes,
    )

    return train_dataset, val_dataset, test_dataset, metadata


def load_german_credit(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the UCI German Credit dataset.

    The German Credit dataset classifies credit risk (good/bad) based on
    financial and demographic features.

    Trunk features (stable): checking_status, credit_history, purpose, savings
    Tail features (variance): credit_amount, duration, age

    Args:
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    import pandas as pd

    # First try local cache (downloaded from UCI)
    # Use relative path
    cache_dir = Path(
        "./data/cache"
    )
    local_file = cache_dir / "german.data"

    # UCI German Credit column names (from german.doc)
    uci_columns = [
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
        "class",
    ]

    if local_file.exists():
        # UCI format: space-delimited, no header
        df = pd.read_csv(local_file, sep=r"\s+", header=None, names=uci_columns)
        target_column = "class"
        # Target is 1=good, 2=bad in UCI format; convert to 0/1
        df["class"] = df["class"].map({1: "good", 2: "bad"})
    else:
        # Fall back to OpenML
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as e:
            raise ImportError(
                "sklearn is required for loading German Credit dataset. "
                "Install with: pip install scikit-learn"
            ) from e

        _setup_fwdproxy()

        # Fetch from OpenML (German Credit dataset)
        german = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        df = german.frame
        target_column = "class"

    # Define feature groups
    continuous_columns = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    categorical_columns = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
    ]

    # Filter to available columns
    continuous_columns = [c for c in continuous_columns if c in df.columns]
    categorical_columns = [c for c in categorical_columns if c in df.columns]

    # Handle missing values
    df = df.dropna()

    # Encode target (good=0, bad=1)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_column].values)
    class_names = list(target_encoder.classes_)

    # Process continuous features
    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Process categorical features
    categorical_encoders: list[LabelEncoder] = []
    categorical_data_list: list[np.ndarray] = []
    category_sizes: list[int] = []

    for col in categorical_columns:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(df[col].values.astype(str))
        categorical_data_list.append(encoded)
        categorical_encoders.append(encoder)
        category_sizes.append(len(encoder.classes_))

    categorical_data = np.column_stack(categorical_data_list).astype(np.int64)

    # Train/val/test split
    X_cont_train, X_cont_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(
            continuous_data,
            categorical_data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_cont_train,
        X_cat_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Create datasets
    train_dataset = TabularDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = TabularDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = TabularDataset(X_cont_test, X_cat_test, y_test)

    metadata = TabularDatasetMetadata(
        name="german_credit",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=len(categorical_columns),
        n_classes=len(class_names),
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=category_sizes,
    )

    return train_dataset, val_dataset, test_dataset, metadata


def load_higgs(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = 500000,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the HIGGS dataset.

    The HIGGS dataset is a binary classification task to distinguish between
    a signal process (Higgs boson) and background processes. ~11M samples.

    Args:
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.
        max_samples: Limit on samples (default 500K for initial experiments).
            Set to None to use all 11M samples.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    import gzip

    import pandas as pd

    # Local cache (downloaded from UCI)
    cache_dir = Path(
        "./data/cache"
    )
    local_file = cache_dir / "HIGGS.csv.gz"

    if not local_file.exists():
        # Try Manifold raw cache (for MAST where local files don't exist)
        tmp_file = Path("/tmp/raw_cache/HIGGS.csv.gz")
        if not tmp_file.exists():
            if not _fetch_raw_from_manifold("HIGGS.csv.gz", tmp_file):
                raise FileNotFoundError(
                    f"HIGGS dataset not found at {local_file} or on Manifold. "
                    "Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
                )
        local_file = tmp_file

    # HIGGS column names (target + 28 features)
    higgs_columns = ["label"] + [f"feature_{i}" for i in range(28)]

    # Read gzipped CSV
    with gzip.open(local_file, "rt") as f:
        if max_samples is not None:
            df = pd.read_csv(f, header=None, names=higgs_columns, nrows=max_samples)
        else:
            df = pd.read_csv(f, header=None, names=higgs_columns)

    # All features are continuous
    target_column = "label"
    continuous_columns = [col for col in df.columns if col != target_column]
    categorical_columns: list[str] = []

    # Target is already 0/1
    y = df[target_column].values.astype(np.int64)
    class_names = ["background", "signal"]

    # Process continuous features
    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Train/val/test split
    X_cont_train, X_cont_test, y_train, y_test = train_test_split(
        continuous_data,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_cont_train, X_cont_val, y_train, y_val = train_test_split(
        X_cont_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    train_dataset = TabularDataset(X_cont_train, None, y_train)
    val_dataset = TabularDataset(X_cont_val, None, y_val)
    test_dataset = TabularDataset(X_cont_test, None, y_test)

    metadata = TabularDatasetMetadata(
        name="higgs",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=0,
        n_classes=2,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=[],
    )

    return train_dataset, val_dataset, test_dataset, metadata


def _load_csv_continuous(
    dataset_label: str,
    local_file: Path,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Generic loader for all-continuous datasets from local CSV files.

    Args:
        dataset_label: Human-readable name for metadata.
        local_file: Path to local CSV file.
        target_column: Name of target column.
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.
        max_samples: Optional limit on number of samples.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    import pandas as pd

    csv_path = local_file
    if not csv_path.exists():
        # Try Manifold raw_cache fallback (for MAST where local files don't exist)
        tmp_path = Path(f"/tmp/raw_cache/{local_file.name}")
        if tmp_path.exists():
            csv_path = tmp_path
        elif _fetch_raw_from_manifold(local_file.name, tmp_path):
            csv_path = tmp_path
        else:
            raise FileNotFoundError(
                f"{dataset_label} dataset not found at {local_file} "
                f"and not available on Manifold raw_cache ({local_file.name})."
            )

    df = pd.read_csv(csv_path)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    continuous_columns = [col for col in df.columns if col != target_column]
    categorical_columns: list[str] = []

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_column].values)
    n_classes = len(le.classes_)
    class_names = [str(c) for c in le.classes_]

    # Process continuous features (coerce non-numeric to NaN, fill with 0)
    continuous_data = (
        df[continuous_columns]
        .apply(lambda s: s.astype(float, errors="ignore"), axis=0)  # pyre-ignore[6]
        .fillna(0)
        .values.astype(np.float32)
    )
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    # Stratified split (fall back to non-stratified if too few per class)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            continuous_data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            continuous_data,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
        )

    train_dataset = TabularDataset(X_train, None, y_train)
    val_dataset = TabularDataset(X_val, None, y_val)
    test_dataset = TabularDataset(X_test, None, y_test)

    metadata = TabularDatasetMetadata(
        name=dataset_label,
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=0,
        n_classes=n_classes,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=[],
    )

    return train_dataset, val_dataset, test_dataset, metadata


_CACHE_DIR = Path(
    "./data/cache"
)


def load_helena(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Helena dataset (~65K samples, 27 features, 100 classes)."""
    return _load_csv_continuous(
        dataset_label="helena",
        local_file=_CACHE_DIR / "helena.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_jannis(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Jannis dataset (~83K samples, 54 features, 4 classes)."""
    return _load_csv_continuous(
        dataset_label="jannis",
        local_file=_CACHE_DIR / "jannis.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_year_prediction(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Year Prediction MSD dataset (~515K samples, 90 features).

    Originally regression; we bin into 5 decade classes for classification.
    """
    import zipfile

    import pandas as pd

    zip_file = _CACHE_DIR / "YearPredictionMSD.txt.zip"
    if not zip_file.exists():
        raise FileNotFoundError(
            f"Year Prediction dataset not found at {zip_file}. "
            "Download from UCI ML repository."
        )

    with zipfile.ZipFile(zip_file) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    # Column 0 is the year (target), rest are features
    target_column = "year"
    df.columns = [target_column] + [f"feature_{i}" for i in range(df.shape[1] - 1)]
    continuous_columns = [col for col in df.columns if col != target_column]
    categorical_columns: list[str] = []

    # Bin years into 5 classes by percentile
    years = df[target_column].astype(float).values
    bins = np.percentile(years, [20, 40, 60, 80])
    y = np.digitize(years, bins).astype(np.int64)
    n_classes = len(np.unique(y))
    class_names = [f"bin_{i}" for i in range(n_classes)]

    continuous_data = df[continuous_columns].values.astype(np.float32)
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(continuous_data)

    X_train, X_test, y_train, y_test = train_test_split(
        continuous_data,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    train_dataset = TabularDataset(X_train, None, y_train)
    val_dataset = TabularDataset(X_val, None, y_val)
    test_dataset = TabularDataset(X_test, None, y_test)

    metadata = TabularDatasetMetadata(
        name="year_prediction",
        n_samples=len(df),
        n_continuous=len(continuous_columns),
        n_categorical=0,
        n_classes=n_classes,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_names=class_names,
        category_sizes=[],
    )

    return train_dataset, val_dataset, test_dataset, metadata


def load_click(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Click prediction dataset (~400K samples, 11 features, binary)."""
    return _load_csv_continuous(
        dataset_label="click",
        local_file=_CACHE_DIR / "click_prediction_small.csv",
        target_column="click",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_aloi(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the ALOI dataset (~108K samples, 128 features, 1000 classes).

    Amsterdam Library of Object Images — 1000-class image feature dataset.
    Widely used in FT-Transformer, TabR, and Grinsztajn et al. (NeurIPS 2022).
    OpenML ID: 42396.
    """
    return _load_csv_continuous(
        dataset_label="aloi",
        local_file=_CACHE_DIR / "aloi.csv",
        target_column="target",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_letter(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Letter Recognition dataset (~20K samples, 16 features, 26 classes).

    Classic UCI dataset for letter recognition.
    Used in SAINT, FT-Transformer, and Grinsztajn et al. (NeurIPS 2022).
    OpenML ID: 6.
    """
    return _load_csv_continuous(
        dataset_label="letter",
        local_file=_CACHE_DIR / "letter.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_dionis(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Dionis dataset (~416K samples, 60 features, 355 classes).

    Large multi-class dataset from AutoML Benchmark Suite 271.
    OpenML ID: 41167.
    """
    return _load_csv_continuous(
        dataset_label="dionis",
        local_file=_CACHE_DIR / "dionis.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_volkert(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Volkert dataset (~58K samples, 180 features, 10 classes).

    Used in AutoML Benchmark Suite 271 and TabPFN v2.
    OpenML ID: 41166.
    """
    return _load_csv_continuous(
        dataset_label="volkert",
        local_file=_CACHE_DIR / "volkert.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_pendigits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Pendigits dataset (~10,992 samples, 16 features, 10 classes).

    Pen-Based Recognition of Handwritten Digits from UCI.
    OpenML ID: 32.
    """
    return _load_csv_continuous(
        dataset_label="pendigits",
        local_file=_CACHE_DIR / "pendigits.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_fashion_mnist(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Fashion-MNIST dataset (~70,000 samples, 784 features, 10 classes).

    Zalando's article images dataset, treated as tabular (flattened pixels).
    OpenML ID: 40996.
    """
    return _load_csv_continuous(
        dataset_label="fashion_mnist",
        local_file=_CACHE_DIR / "fashion_mnist.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_texture(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Texture dataset (~5,500 samples, 40 features, 11 classes).

    Brodatz texture classification dataset.
    OpenML ID: 40499.
    """
    return _load_csv_continuous(
        dataset_label="texture",
        local_file=_CACHE_DIR / "texture.csv",
        target_column="Class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def load_shuttle(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load the Shuttle dataset (~58,000 samples, 9 features, 7 classes).

    NASA Space Shuttle Statlog dataset.
    OpenML ID: 40685.
    """
    return _load_csv_continuous(
        dataset_label="shuttle",
        local_file=_CACHE_DIR / "shuttle.csv",
        target_column="class",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples=max_samples,
    )


def create_dataloaders(
    train_dataset: TabularDataset,
    val_dataset: TabularDataset,
    test_dataset: TabularDataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:  # pyre-ignore[11]
    """Create DataLoaders from TabularDatasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Dataset registry for easy access
DATASET_REGISTRY: dict[str, Any] = {
    "adult": load_adult,
    "bank_marketing": load_bank_marketing,
    "german_credit": load_german_credit,
    "covertype": load_covertype,
    "higgs": load_higgs,
    "helena": load_helena,
    "jannis": load_jannis,
    "year_prediction": load_year_prediction,
    "click": load_click,
    "aloi": load_aloi,
    "letter": load_letter,
    "dionis": load_dionis,
    "volkert": load_volkert,
    "pendigits": load_pendigits,
    "fashion_mnist": load_fashion_mnist,
    "texture": load_texture,
    "shuttle": load_shuttle,
    "synthetic": load_synthetic,
}


def load_dataset(
    name: str,
    **kwargs: Any,
) -> tuple[TabularDataset, TabularDataset, TabularDataset, TabularDatasetMetadata]:
    """Load a dataset by name.

    Tries Manifold cache first (pre-processed .pt snapshots), then falls
    through to local loaders.  Manifold is only used when no custom kwargs
    are passed, since cached files were produced with default parameters.

    Args:
        name: Dataset name (see DATASET_REGISTRY for available names).
        **kwargs: Additional arguments passed to the loader.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).

    Raises:
        ValueError: If dataset name is not recognized.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        )

    # Try Manifold cache when using default parameters.
    if not kwargs:
        result = _load_from_manifold(name)
        if result is not None:
            return result

    return DATASET_REGISTRY[name](**kwargs)
