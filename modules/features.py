

"""Shared feature extraction, selection, and caching utilities.

Provides a common pipeline for all experiment scripts that need raw tabular
features, supervised/unsupervised feature selection, and normalization before
residual quantization (RQ).

Feature selection uses XGBoost feature importance (gain-based) with automatic
knee detection. XGBoost is fast even on large datasets (hundreds of thousands
of samples, hundreds of features, hundreds of classes), handles mixed
continuous/categorical features natively, and captures non-linear relationships
and feature interactions — unlike marginal MI which evaluates features
independently.

Example:
    >>> from modules.features import (
    ...     collect_raw_features,
    ...     select_features_supervised,
    ...     normalize_for_rq,
    ...     FeatureCache,
    ... )
    >>> features, labels = collect_raw_features(train_dataset)
    >>> indices, scores = select_features_supervised(features, labels)
    >>> scaled, _, scaler = normalize_for_rq(features[:, indices])
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import logging
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def collect_raw_features(
    dataset: torch.utils.data.Dataset,  # pyre-ignore[2]
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw features and labels from a dataset as numpy arrays.

    Concatenates continuous features and categorical features (cast to float)
    into a single (N, d_raw) matrix.

    Args:
        dataset: A TabularDataset that yields (features_dict, labels) batches.
        batch_size: Batch size for the DataLoader.

    Returns:
        Tuple of (features, labels) as float32 / int64 numpy arrays.
        features has shape (N, d_raw), labels has shape (N,).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch_features, batch_y in loader:
        parts: list[torch.Tensor] = []
        if isinstance(batch_features, dict):
            cont = batch_features.get("cont_features")
            if cont is not None:
                parts.append(cont)
            cat = batch_features.get("cat_features")
            if cat is not None:
                parts.append(cat.float())
        elif isinstance(batch_features, torch.Tensor):
            parts.append(batch_features.float())
        if not parts:
            raise ValueError(
                f"Cannot extract features from {type(batch_features)}: "
                f"keys={list(batch_features.keys()) if isinstance(batch_features, dict) else 'N/A'}"
            )
        all_features.append(torch.cat(parts, dim=-1))
        all_labels.append(batch_y)

    features = torch.cat(all_features).numpy().astype(np.float32)
    labels = torch.cat(all_labels).numpy()
    return features, labels


def _find_knee(scores: np.ndarray) -> int:
    """Find the knee/elbow point in a sorted descending score array.

    Uses the maximum-distance-from-diagonal method: normalize x and y
    to [0, 1], then find the index with maximum perpendicular distance
    from the line connecting the first and last points.

    Args:
        scores: 1-D array of scores sorted in descending order.

    Returns:
        Number of features to keep (1-indexed: returns k such that
        keeping the top-k features is optimal).
    """
    n = len(scores)
    if n <= 2:
        return n

    # Normalize to [0, 1] range
    x = np.linspace(0, 1, n)
    y_min, y_max = scores[-1], scores[0]
    if y_max - y_min < 1e-12:
        # All scores are the same; keep all
        return n
    y = (scores - y_min) / (y_max - y_min)

    # Line from first to last point
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    line_vec = p1 - p0
    line_len = np.sqrt(line_vec @ line_vec)
    if line_len < 1e-12:
        return n
    line_unit = line_vec / line_len

    # Perpendicular distance from each point to the line
    dists = np.abs(np.cross(line_unit, p0 - np.column_stack([x, y])))

    # The knee is the point with maximum distance
    knee_idx = int(np.argmax(dists))

    # Return k (1-indexed): keep features 0..knee_idx inclusive
    return max(knee_idx + 1, 1)


def select_features_supervised(
    features: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select features using XGBoost feature importance (gain-based).

    Trains a quick XGBoost classifier, extracts gain-based feature importance,
    and uses the knee/elbow method to determine how many features to keep.

    XGBoost importance captures non-linear relationships and feature
    interactions, unlike marginal MI which evaluates features independently.
    It's also fast on large datasets (O(N log N) per tree).

    Args:
        features: (N, d) float32 feature matrix.
        labels: (N,) integer class labels.
        k: Number of features to keep. If None, determined by knee method.
        random_state: Random state for XGBoost.

    Returns:
        Tuple of (selected_indices, importance_scores):
            - selected_indices: (k,) int array of column indices to keep,
              sorted in descending importance order.
            - importance_scores: (d,) float array of importance for all features.
    """
    import xgboost as xgb  # pyre-ignore[21]

    n_classes = len(set(labels.tolist()))
    n_samples = features.shape[0]

    # Quick shallow XGBoost — enough to rank features, not to maximize accuracy
    params: dict[str, object] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": min(1.0, 10000 / max(n_samples, 1)),  # cap subsample for speed
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
        "importance_type": "gain",
    }
    if n_classes <= 2:
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **params,  # pyre-ignore[6]
        )
    else:
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            **params,  # pyre-ignore[6]
        )

    clf.fit(features, labels)
    importance_scores = clf.feature_importances_.astype(np.float64)

    logger.info(
        f"  XGBoost feature importance: {features.shape[1]} features, "
        f"{n_classes} classes, top-5 importance: "
        f"{sorted(importance_scores, reverse=True)[:5]}"
    )

    # Sort by importance descending
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_scores = importance_scores[sorted_indices]

    if k is None:
        k = _find_knee(sorted_scores)

    k = min(k, features.shape[1])
    k = max(k, 1)
    selected = sorted_indices[:k]

    logger.info(f"  Selected {k} of {features.shape[1]} features (knee method)")

    return selected, importance_scores


def select_features_unsupervised(
    features: np.ndarray,
    k: Optional[int] = None,
    variance_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Select features without labels using variance-based filtering.

    Drops near-zero-variance features, then uses the knee method on
    sorted variances to determine how many to keep (if k is not given).

    Args:
        features: (N, d) float32 feature matrix.
        k: Number of features to keep. If None, determined by knee method.
        variance_threshold: Minimum variance to be considered non-constant.

    Returns:
        Tuple of (selected_indices, variances):
            - selected_indices: (k,) int array of column indices to keep,
              sorted in descending variance order.
            - variances: (d,) float array of per-feature variances.
    """
    variances = np.var(features, axis=0).astype(np.float64)

    # Filter out near-zero variance features
    nonconst_mask = variances > variance_threshold
    if not np.any(nonconst_mask):
        # Degenerate case: all features are constant; keep all
        return np.arange(features.shape[1]), variances

    # Sort non-constant features by variance descending
    nonconst_indices = np.where(nonconst_mask)[0]
    nonconst_vars = variances[nonconst_indices]
    order = np.argsort(nonconst_vars)[::-1]
    sorted_indices = nonconst_indices[order]
    sorted_vars = nonconst_vars[order]

    if k is None:
        k = _find_knee(sorted_vars)

    k = min(k, len(sorted_indices))
    k = max(k, 1)
    selected = sorted_indices[:k]
    return selected, variances


def normalize_for_rq(
    train_features: np.ndarray,
    test_features: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Standardize features before RQ fitting.

    Applies StandardScaler (zero mean, unit variance) to the training
    features and optionally transforms test features with the same scaler.

    Args:
        train_features: (N_train, d) float32 feature matrix.
        test_features: Optional (N_test, d) float32 feature matrix.

    Returns:
        Tuple of (train_scaled, test_scaled, scaler):
            - train_scaled: (N_train, d) float32 standardized features.
            - test_scaled: (N_test, d) float32 or None.
            - scaler: Fitted StandardScaler for later inverse_transform.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = None
    if test_features is not None:
        test_scaled = scaler.transform(test_features).astype(np.float32)
    return train_scaled, test_scaled, scaler


class FeatureCache:
    """Persistent cache for feature extraction and selection results.

    Stores artifacts as .npz (numpy) and .json (metadata) files under
    a cache directory, keyed by dataset name and parameters.

    Example:
        >>> cache = FeatureCache("results/feature_cache")
        >>> features, labels = cache.get_or_compute_raw_features(
        ...     "adult", lambda: collect_raw_features(train_dataset)
        ... )
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset: str) -> Path:
        d = self.cache_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _hash_params(**kwargs: object) -> str:
        """Create a short hash from arbitrary parameters."""
        key_str = json.dumps(
            {k: str(v) for k, v in sorted(kwargs.items())}, sort_keys=True
        )
        return hashlib.sha256(key_str.encode()).hexdigest()[:12]

    # -- Raw features --

    def get_or_compute_raw_features(
        self,
        dataset: str,
        compute_fn: object,  # Callable[[], tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get cached raw features or compute and cache them.

        Args:
            dataset: Dataset name.
            compute_fn: Zero-arg callable returning (features, labels).

        Returns:
            Tuple of (features, labels) as numpy arrays.
        """
        path = self._dataset_dir(dataset) / "raw_features.npz"
        if path.exists():
            data = np.load(path)
            return data["features"], data["labels"]

        features, labels = compute_fn()  # pyre-ignore[29]
        np.savez(path, features=features, labels=labels)
        return features, labels

    # -- Feature selection --

    def get_or_compute_selection(
        self,
        dataset: str,
        method: str,
        compute_fn: object,  # Callable[[], tuple[np.ndarray, np.ndarray]]
        k: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get cached feature selection results or compute and cache them.

        Args:
            dataset: Dataset name.
            method: Selection method name ("supervised" or "unsupervised").
            compute_fn: Zero-arg callable returning (indices, scores).
            k: Number of features requested (None = auto).

        Returns:
            Tuple of (selected_indices, scores) as numpy arrays.
        """
        k_str = str(k) if k is not None else "auto"
        path = self._dataset_dir(dataset) / f"selection_{method}_k{k_str}.npz"
        if path.exists():
            data = np.load(path)
            return data["indices"], data["scores"]

        indices, scores = compute_fn()  # pyre-ignore[29]
        np.savez(path, indices=indices, scores=scores)
        return indices, scores

    # -- D_cut search results --

    def get_dcut_results(
        self,
        dataset: str,
        feature_mode: str,
        nbits: int,
    ) -> Optional[dict[str, object]]:
        """Load cached d_cut search results.

        Args:
            dataset: Dataset name.
            feature_mode: "embeddings", "raw", or "selected".
            nbits: Number of bits per RQ level.

        Returns:
            Dict with d_cut results (optimal_dcut, entropy_curve, etc.)
            or None if not cached.
        """
        path = self._dataset_dir(dataset) / f"dcut_{feature_mode}_nbits{nbits}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)  # pyre-ignore[7]
        return None

    def save_dcut_results(
        self,
        dataset: str,
        feature_mode: str,
        nbits: int,
        results: dict[str, object],
    ) -> None:
        """Save d_cut search results to cache.

        Args:
            dataset: Dataset name.
            feature_mode: "embeddings", "raw", or "selected".
            nbits: Number of bits per RQ level.
            results: Dict with d_cut results to cache.
        """
        path = self._dataset_dir(dataset) / f"dcut_{feature_mode}_nbits{nbits}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
