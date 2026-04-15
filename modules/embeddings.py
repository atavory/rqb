

"""Shared embedding extraction for fi_trunk_tail experiments.

Provides a unified ``EmbeddingExtractor`` that converts raw tabular features
into one of three representation spaces:

* **raw** — identity (returns the input features unchanged).
* **lgbm** — supervised class-probability embeddings from a LightGBM
  classifier.  ``predict_proba(X)`` returns ``(N, n_classes)`` probability
  vectors that live on a simplex and encode the model's learned
  discriminative structure.
* **iforest** — unsupervised anomaly-score embeddings from an Isolation
  Forest.  Per-tree normalised path lengths give an ``(N, n_estimators)``
  continuous embedding that captures density structure without labels.

All outputs are ``float32`` numpy arrays ready for ``StandardScaler`` + FAISS RQ.

Example::

    >>> from modules.embeddings import (
    ...     EmbeddingConfig,
    ...     EmbeddingExtractor,
    ... )
    >>> cfg = EmbeddingConfig(embedding_type="lgbm", n_estimators=200)
    >>> extractor = EmbeddingExtractor(cfg)
    >>> extractor.fit(X_train, y_train)
    >>> Z_train = extractor.transform(X_train)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction.

    Attributes:
        embedding_type: One of ``"raw"``, ``"lgbm"``, or ``"iforest"``.
        n_estimators: Number of trees for lgbm / iforest.
        random_state: Random seed for reproducibility.
    """

    embedding_type: str = "raw"
    n_estimators: int = 200
    random_state: int = 42


class EmbeddingExtractor:
    """Extract embeddings from raw tabular features.

    Args:
        config: An :class:`EmbeddingConfig` specifying the embedding type and
            hyper-parameters.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: object = None  # LGBMClassifier or IsolationForest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None) -> None:
        """Fit the embedding model on training data.

        For ``"raw"`` this is a no-op.  ``"lgbm"`` requires labels
        (``y_train``); ``"iforest"`` ignores them.

        Args:
            X_train: Training features, shape ``(N, d)``.
            y_train: Training labels, shape ``(N,)``.  Required for lgbm.
        """
        if self.config.embedding_type == "raw":
            return

        if self.config.embedding_type == "lgbm":
            self._fit_lgbm(X_train, y_train)
        elif self.config.embedding_type == "iforest":
            self._fit_iforest(X_train)
        else:
            raise ValueError(
                f"Unknown embedding_type: {self.config.embedding_type!r}. "
                f"Must be one of 'raw', 'lgbm', 'iforest'."
            )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features into the chosen embedding space.

        Args:
            X: Feature matrix, shape ``(N, d)``.

        Returns:
            Embedding matrix as ``float32`` with shape ``(N, embed_dim)``.
        """
        if self.config.embedding_type == "raw":
            return X.astype(np.float32)

        if self.config.embedding_type == "lgbm":
            return self._transform_lgbm(X)
        elif self.config.embedding_type == "iforest":
            return self._transform_iforest(X)
        else:
            raise ValueError(
                f"Unknown embedding_type: {self.config.embedding_type!r}."
            )

    def fit_transform(
        self, X_train: np.ndarray, y_train: np.ndarray | None = None
    ) -> np.ndarray:
        """Convenience wrapper: ``fit`` then ``transform``."""
        self.fit(X_train, y_train)
        return self.transform(X_train)

    # ------------------------------------------------------------------
    # LightGBM helpers
    # ------------------------------------------------------------------

    def _fit_lgbm(self, X: np.ndarray, y: np.ndarray | None) -> None:
        import lightgbm as lgb  # pyre-ignore[21]

        if y is None:
            raise ValueError("y_train is required for lgbm embedding type.")

        clf = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            verbosity=-1,
        )
        clf.fit(X, y)
        self._model = clf

    def _transform_lgbm(self, X: np.ndarray) -> np.ndarray:
        import lightgbm as lgb  # pyre-ignore[21]  # noqa: F811

        model: lgb.LGBMClassifier = self._model  # pyre-ignore[8]
        proba = model.predict_proba(X)  # pyre-ignore[16]
        # predict_proba returns (N, n_classes) probability vectors.
        return np.asarray(proba, dtype=np.float32)

    # ------------------------------------------------------------------
    # Isolation Forest helpers
    # ------------------------------------------------------------------

    def _fit_iforest(self, X: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
        )
        iso.fit(X)
        self._model = iso

    def _transform_iforest(self, X: np.ndarray) -> np.ndarray:
        from sklearn.ensemble import IsolationForest

        iso: IsolationForest = self._model  # pyre-ignore[8]
        # Per-tree normalised path lengths: continuous (N, n_estimators)
        # embedding capturing density structure.
        paths: list[np.ndarray] = []
        for tree, features in zip(
            iso.estimators_, iso.estimators_features_  # pyre-ignore[16]
        ):
            # decision_path gives a sparse (N, n_nodes) indicator matrix.
            # The number of edges traversed = number of non-zeros per row - 1
            # (root node is always visited).  Normalise by max possible depth.
            indicator = tree.decision_path(X[:, features])  # pyre-ignore[16]
            # Path length per sample (number of nodes visited).
            path_lengths = np.asarray(indicator.sum(axis=1)).ravel().astype(np.float32)
            # Normalise by tree depth so all trees are on comparable scale.
            max_depth = tree.get_depth()  # pyre-ignore[16]
            if max_depth > 0:
                path_lengths = path_lengths / max_depth
            paths.append(path_lengths)
        return np.column_stack(paths)
