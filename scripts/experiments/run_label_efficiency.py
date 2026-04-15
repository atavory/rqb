#!/usr/bin/env python3


"""Label efficiency experiment: trunk codes as a labeling oracle.

Demonstrates that RQ trunk codes enable extreme label efficiency:
labeling ONE point per trunk code bucket suffices to classify the
entire dataset, because trunk codes are label-pure.

Protocol:
    1. Train encoder on full labeled train set (simulates having a
       pre-trained representation; labels are used ONLY here).
    2. Fit RQ on train embeddings (unsupervised — uses no labels).
    3. "Forget" all labels. Now select which points to label using
       different strategies, with a budget of B labels.
    4. Train a classifier on ONLY the B labeled points.
    5. Evaluate on held-out test set.

Labeling strategies:
    - Random: sample B points uniformly at random
    - Uncertainty: train initial classifier on small random seed,
      iteratively label the most uncertain points
    - CoreSet: greedy k-center in embedding space (Sener & Savarese 2018)
    - BADGE: gradient embedding diversity via k-means++ (Ash et al. 2020)
    - RQ-Stratified: sample 1 per trunk code bucket (B = n_buckets),
      then fill remaining budget from largest buckets
    - KM-Stratified: same but with KMeans clusters
    - RQ-Propagate: label 1 per bucket, assign all bucket members
      the same label (zero additional classifier training)
    - KM-Propagate: same with KMeans clusters

Metrics:
    - Test accuracy vs number of labels (learning curve)
    - Labels needed to reach X% of full-data accuracy (efficiency)

Example:
    python3 scripts/experiments/run_label_efficiency.py -- \\
        --dataset adult --n-encoder-seeds 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import logging
from modules.data import DATASET_REGISTRY, load_dataset
from modules.embeddings import (
    EmbeddingConfig,
    EmbeddingExtractor,
)
from modules.encoders.tab_transformer import MLPEncoder
from modules.features import (
    collect_raw_features,
    FeatureCache,
    normalize_for_rq,
    select_features_supervised,
)
from modules.significance import (
    compute_pairwise_significance,
    log_significance,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Duplicated helpers (self-contained python_binary pattern)
# =============================================================================


def get_features(
    batch_features: dict[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor | None]:
    """Extract continuous and categorical features from batch."""
    cont_features = batch_features["cont_features"].to(device)
    cat_features = batch_features.get("cat_features")
    if cat_features is not None:
        cat_features = cat_features.to(device)
    return cont_features, cat_features


def pretrain_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,  # pyre-ignore[11]
    n_classes: int,
    embedding_dim: int,
    lr: float,
    weight_decay: float,
    encoder_epochs: int,
    device: torch.device,
) -> nn.Module:
    """Pre-train encoder with classification loss."""
    classifier = nn.Linear(embedding_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    encoder.train()
    for epoch in range(encoder_epochs):
        total_loss = 0.0
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            batch_y = batch_y.to(device)

            embeddings = encoder(cat_features=cat_features, cont_features=cont_features)
            logits = classifier(embeddings)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f"  Encoder epoch {epoch + 1}: loss={avg_loss:.4f}")

    return encoder


def collect_embeddings_and_labels(
    encoder: nn.Module,
    loader: DataLoader,  # pyre-ignore[11]
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect all embeddings and labels from a data loader."""
    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for batch_features, batch_y in loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_embeddings.append(emb.cpu())
            all_labels.append(batch_y)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def conditional_entropy(
    labels: np.ndarray,
    codes: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Compute H(Y | Z) using plug-in estimator with add-alpha smoothing."""
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)

    code_keys = [tuple(c) for c in codes]
    n = len(labels)
    unique_labels = set(labels.tolist())
    n_labels = len(unique_labels)

    joint_counts: dict[tuple[tuple[int, ...], int], int] = {}
    code_counts: dict[tuple[int, ...], int] = {}

    for code_key, label in zip(code_keys, labels.tolist()):
        joint_counts[(code_key, label)] = joint_counts.get((code_key, label), 0) + 1
        code_counts[code_key] = code_counts.get(code_key, 0) + 1

    cond_entropy = 0.0
    for code_key, code_count in code_counts.items():
        p_z = code_count / n
        h_y_given_z = 0.0
        total_smoothed = code_count + alpha * n_labels
        for label in unique_labels:
            jc = joint_counts.get((code_key, label), 0)
            p = (jc + alpha) / total_smoothed
            if p > 0:
                h_y_given_z -= p * math.log(p)
        cond_entropy += p_z * h_y_given_z

    return cond_entropy


# =============================================================================
# Config
# =============================================================================


@dataclass
class LabelEfficiencyConfig:
    """Configuration for label efficiency experiment."""

    dataset: str = "adult"
    n_encoder_seeds: int = 3
    n_label_seeds: int = 5  # Random seeds for label selection
    seed: int = 42

    # Encoder
    embedding_dim: int = 64
    hidden_dim: int = 128
    encoder_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256

    # RQ
    nbits: int = 6
    d_cut_values: list[int] = field(default_factory=lambda: [1, 2, 3])
    auto_dcut: bool = True

    # Label budgets (number of labeled points)
    label_budgets: list[int] = field(
        default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    )

    device: str = "cuda"

    # Feature mode: "embeddings" (default), "raw", "selected", "lgbm", "iforest"
    feature_mode: str = "embeddings"
    # Cache directory for raw features and feature selection results
    feature_cache_dir: str = "results/feature_cache"
    # Embedding type: "raw" (identity), "lgbm" (leaf scores), "iforest" (path depths)
    embedding_type: str = "raw"

    # Auto d_cut: override nbits/d_cut from per-dataset configs
    auto_dcut: bool = False


# =============================================================================
# Labeling strategies
# =============================================================================


def _train_and_eval(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    selected_idx: np.ndarray,
) -> float:
    """Train logistic regression on selected subset, evaluate on test."""
    X = train_emb[selected_idx]
    y = train_labels[selected_idx]

    # Need at least 2 classes in the labeled set
    if len(np.unique(y)) < 2:
        # Fall back to majority class prediction
        majority = Counter(y.tolist()).most_common(1)[0][0]
        return float((test_labels == majority).mean())

    clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    clf.fit(X, y)
    return float(clf.score(test_emb, test_labels))


def _propagate_and_eval(
    train_labels: np.ndarray,
    test_codes: np.ndarray,
    bucket_to_label: dict[tuple[int, ...], int],
    test_labels: np.ndarray,
    n_classes: int,
    rng: np.random.RandomState,
) -> float:
    """Classify test points by propagating bucket labels. No classifier."""
    n_test = len(test_labels)
    preds = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        key = tuple(test_codes[i].tolist())
        if key in bucket_to_label:
            preds[i] = bucket_to_label[key]
        else:
            # Unseen bucket: random class
            preds[i] = int(rng.randint(0, n_classes))
    return float((preds == test_labels).mean())


def strategy_random(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """Random sampling baseline."""
    n = train_emb.shape[0]
    selected = rng.choice(n, min(budget, n), replace=False)
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, selected)


def strategy_stratified(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    codes: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """Stratified sampling: 1 per bucket, then fill from largest buckets."""
    # Build buckets
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(len(codes)):
        key = tuple(codes[i].tolist())
        buckets[key].append(i)

    selected: list[int] = []

    # Phase 1: one per bucket
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    for key in bucket_keys:
        if len(selected) >= budget:
            break
        idx = rng.choice(buckets[key])
        selected.append(idx)

    # Phase 2: fill remaining budget from largest buckets
    if len(selected) < budget:
        # Sort buckets by size, sample more from bigger ones
        sorted_buckets = sorted(buckets.items(), key=lambda x: -len(x[1]))
        remaining = budget - len(selected)
        selected_set = set(selected)
        for key, members in sorted_buckets:
            if remaining <= 0:
                break
            available = [m for m in members if m not in selected_set]
            if available:
                n_pick = min(len(available), max(1, remaining // len(sorted_buckets)))
                n_pick = min(n_pick, remaining)
                picks = rng.choice(available, n_pick, replace=False)
                selected.extend(picks.tolist())
                selected_set.update(picks.tolist())
                remaining = budget - len(selected)

    selected_arr = np.array(selected[:budget], dtype=np.int64)
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, selected_arr)


def strategy_propagate(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    n_classes: int,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """Label propagation: label 1 per bucket, propagate to all members."""
    # Build buckets
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(len(train_codes)):
        key = tuple(train_codes[i].tolist())
        buckets[key].append(i)

    # Label one point per bucket (up to budget)
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    bucket_to_label: dict[tuple[int, ...], int] = {}
    n_labeled = 0
    for key in bucket_keys:
        if n_labeled >= budget:
            break
        idx = rng.choice(buckets[key])
        bucket_to_label[key] = int(train_labels[idx])
        n_labeled += 1

    return _propagate_and_eval(
        train_labels, test_codes, bucket_to_label, test_labels, n_classes, rng
    )


def strategy_uncertainty(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """Uncertainty sampling: iteratively label most uncertain points."""
    n = train_emb.shape[0]
    n_classes = len(np.unique(train_labels))

    # Start with a small random seed (10 points or budget//5)
    seed_size = min(max(10, n_classes * 2), budget)
    labeled = set(rng.choice(n, seed_size, replace=False).tolist())

    # Iteratively add most uncertain points
    batch_add = max(1, (budget - len(labeled)) // 10)

    while len(labeled) < budget:
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)
        X_labeled = train_emb[labeled_arr]
        y_labeled = train_labels[labeled_arr]

        if len(np.unique(y_labeled)) < 2:
            # Not enough classes yet, add random
            unlabeled = list(set(range(n)) - labeled)
            add_n = min(batch_add, budget - len(labeled), len(unlabeled))
            new_idx = rng.choice(unlabeled, add_n, replace=False)
            labeled.update(new_idx.tolist())
            continue

        clf = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
        clf.fit(X_labeled, y_labeled)

        # Get uncertainty for unlabeled points
        unlabeled = np.array(sorted(set(range(n)) - labeled), dtype=np.int64)
        if len(unlabeled) == 0:
            break
        probs = clf.predict_proba(train_emb[unlabeled])
        # Entropy-based uncertainty
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        add_n = min(batch_add, budget - len(labeled), len(unlabeled))
        most_uncertain = np.argpartition(entropy, -add_n)[-add_n:]
        labeled.update(unlabeled[most_uncertain].tolist())

    labeled_arr = np.array(sorted(labeled), dtype=np.int64)[:budget]
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, labeled_arr)


def strategy_coreset(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """CoreSet baseline (Sener & Savarese, ICLR 2018): greedy k-center.

    Iteratively selects the point with the maximum minimum distance to all
    already-selected points, producing a diverse coreset in embedding space.
    """
    n = train_emb.shape[0]
    actual_budget = min(budget, n)

    # Start with 1 random point
    selected: list[int] = [int(rng.randint(0, n))]

    # Precompute distances from first selected point to all points
    # min_dist[i] = min distance from point i to any selected point
    diff = train_emb - train_emb[selected[0]]
    min_dist = np.sum(diff * diff, axis=1)  # squared L2

    for _ in range(1, actual_budget):
        # Select point with maximum min-distance to selected set
        farthest = int(np.argmax(min_dist))
        selected.append(farthest)

        # Update min distances
        diff = train_emb - train_emb[farthest]
        new_dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    selected_arr = np.array(selected, dtype=np.int64)
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, selected_arr)


def strategy_badge(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """BADGE baseline (Ash et al., ICLR 2020): gradient embedding diversity.

    Trains an initial classifier on a small random seed, computes gradient
    embeddings for unlabeled points (gradient of the loss w.r.t. last-layer
    parameters using predicted class), then selects a diverse batch via
    k-means++ initialization on the gradient embeddings.
    """
    from sklearn.cluster import kmeans_plusplus
    from sklearn.decomposition import PCA as SklearnPCA

    n = train_emb.shape[0]
    n_classes = len(np.unique(train_labels))
    dim = train_emb.shape[1]
    grad_dim = n_classes * dim

    # Scalability constants: sklearn kmeans_plusplus is O(k^2 * n * d),
    # so we cap batch size, subsample candidates, and reduce dimensions.
    MAX_BATCH = 100  # cap per-iteration k-means++ centers
    MAX_CANDIDATES = 10000  # subsample unlabeled pool for k-means++
    MAX_GRAD_DIM = 256  # PCA target when grad_dim is large

    # Start with a small random seed (same logic as uncertainty strategy)
    seed_size = min(max(10, n_classes * 2), budget)
    labeled = set(rng.choice(n, seed_size, replace=False).tolist())

    # Iterative batch selection via gradient embeddings
    batch_add = min(MAX_BATCH, max(1, (budget - len(labeled)) // 5))

    while len(labeled) < budget:
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)
        X_labeled = train_emb[labeled_arr]
        y_labeled = train_labels[labeled_arr]

        if len(np.unique(y_labeled)) < 2:
            # Not enough classes yet, add random
            unlabeled = list(set(range(n)) - labeled)
            add_n = min(batch_add, budget - len(labeled), len(unlabeled))
            new_idx = rng.choice(unlabeled, add_n, replace=False)
            labeled.update(new_idx.tolist())
            continue

        clf = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
        clf.fit(X_labeled, y_labeled)

        unlabeled = np.array(sorted(set(range(n)) - labeled), dtype=np.int64)
        if len(unlabeled) == 0:
            break

        # Compute gradient embeddings for unlabeled points
        X_unlabeled = train_emb[unlabeled]
        probs = clf.predict_proba(X_unlabeled)  # (n_unlabeled, n_classes)
        preds = np.argmax(probs, axis=1)

        # Construct gradient embedding: for each point, the gradient is
        # [p1*x, p2*x, ..., (p_y - 1)*x, ...] where y is the predicted class.
        # We use the "hypothesized label" trick from BADGE.
        grad_emb = np.zeros((len(unlabeled), grad_dim), dtype=np.float32)
        for c in range(n_classes):
            mask = preds == c
            if not np.any(mask):
                continue
            p = probs[:, c].copy()
            p[mask] -= 1.0  # subtract 1 for predicted class
            grad_emb[:, c * dim : (c + 1) * dim] = p[:, np.newaxis] * X_unlabeled

        # Dimensionality reduction when gradient space is large
        # (e.g. helena: 100 classes × 64 dim = 6400-d)
        if grad_dim > MAX_GRAD_DIM:
            n_components = min(MAX_GRAD_DIM, len(unlabeled) - 1)
            pca = SklearnPCA(n_components=n_components, random_state=42)
            grad_emb = pca.fit_transform(grad_emb)

        # Subsample candidate pool for k-means++ when pool is large
        add_n = min(batch_add, budget - len(labeled), len(unlabeled))
        if len(unlabeled) > MAX_CANDIDATES:
            sub_idx = rng.choice(len(unlabeled), MAX_CANDIDATES, replace=False)
            grad_sub = grad_emb[sub_idx]
            try:
                _, sel_in_sub = kmeans_plusplus(
                    grad_sub,
                    n_clusters=add_n,
                    random_state=int(rng.randint(0, 2**31)),
                )
                new_points = unlabeled[sub_idx[sel_in_sub]].tolist()
            except Exception:
                new_points = rng.choice(unlabeled, add_n, replace=False).tolist()
        else:
            try:
                _, indices = kmeans_plusplus(
                    grad_emb,
                    n_clusters=add_n,
                    random_state=int(rng.randint(0, 2**31)),
                )
                new_points = unlabeled[indices].tolist()
            except Exception:
                new_points = rng.choice(unlabeled, add_n, replace=False).tolist()
        labeled.update(new_points)

    labeled_arr = np.array(sorted(labeled), dtype=np.int64)[:budget]
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, labeled_arr)


def strategy_rq_seeded_uncertainty(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    codes: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """RQ-seeded uncertainty: stratified init, then uncertainty sampling.

    Uses RQ-stratified sampling to pick the initial seed set (one per bucket),
    ensuring coverage of all regions. Then switches to standard uncertainty
    sampling to refine near decision boundaries.
    """
    n = train_emb.shape[0]

    # Build buckets for stratified seeding
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(len(codes)):
        key = tuple(codes[i].tolist())
        buckets[key].append(i)

    # Phase 1: one point per bucket as seed
    labeled: set[int] = set()
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    for key in bucket_keys:
        if len(labeled) >= budget:
            break
        idx = rng.choice(buckets[key])
        labeled.add(int(idx))

    if len(labeled) >= budget:
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)[:budget]
        return _train_and_eval(
            train_emb, train_labels, test_emb, test_labels, labeled_arr
        )

    # Phase 2: uncertainty sampling for remaining budget
    batch_add = max(1, (budget - len(labeled)) // 10)

    while len(labeled) < budget:
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)
        X_labeled = train_emb[labeled_arr]
        y_labeled = train_labels[labeled_arr]

        if len(np.unique(y_labeled)) < 2:
            unlabeled = list(set(range(n)) - labeled)
            add_n = min(batch_add, budget - len(labeled), len(unlabeled))
            new_idx = rng.choice(unlabeled, add_n, replace=False)
            labeled.update(new_idx.tolist())
            continue

        clf = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
        clf.fit(X_labeled, y_labeled)

        unlabeled = np.array(sorted(set(range(n)) - labeled), dtype=np.int64)
        if len(unlabeled) == 0:
            break
        probs = clf.predict_proba(train_emb[unlabeled])
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        add_n = min(batch_add, budget - len(labeled), len(unlabeled))
        most_uncertain = np.argpartition(entropy, -add_n)[-add_n:]
        labeled.update(unlabeled[most_uncertain].tolist())

    labeled_arr = np.array(sorted(labeled), dtype=np.int64)[:budget]
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, labeled_arr)


def strategy_rq_diverse_uncertainty(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    codes: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """RQ-diverse uncertainty: uncertainty sampling with per-bucket diversity.

    Like uncertainty sampling, but each batch picks the most uncertain point
    from each trunk code bucket (instead of the globally most uncertain, which
    may cluster in one region). Ensures spatial diversity among queried points.
    """
    n = train_emb.shape[0]
    n_classes = len(np.unique(train_labels))

    # Build bucket membership
    point_to_bucket: np.ndarray = np.array(
        [tuple(codes[i].tolist()) for i in range(n)], dtype=object
    )

    # Start with a small stratified seed (one per bucket, up to seed_size)
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(n):
        buckets[tuple(codes[i].tolist())].append(i)

    seed_size = min(max(10, n_classes * 2), budget)
    labeled: set[int] = set()
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    for key in bucket_keys:
        if len(labeled) >= seed_size:
            break
        idx = rng.choice(buckets[key])
        labeled.add(int(idx))
    # Fill to seed_size if fewer buckets than seed_size
    if len(labeled) < seed_size:
        unlabeled = list(set(range(n)) - labeled)
        add_n = min(seed_size - len(labeled), len(unlabeled))
        labeled.update(rng.choice(unlabeled, add_n, replace=False).tolist())

    # Iteratively add most uncertain per bucket
    while len(labeled) < budget:
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)
        X_labeled = train_emb[labeled_arr]
        y_labeled = train_labels[labeled_arr]

        if len(np.unique(y_labeled)) < 2:
            unlabeled = list(set(range(n)) - labeled)
            add_n = min(1, budget - len(labeled), len(unlabeled))
            labeled.update(rng.choice(unlabeled, add_n, replace=False).tolist())
            continue

        clf = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
        clf.fit(X_labeled, y_labeled)

        unlabeled = np.array(sorted(set(range(n)) - labeled), dtype=np.int64)
        if len(unlabeled) == 0:
            break
        probs = clf.predict_proba(train_emb[unlabeled])
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # Pick most uncertain per bucket
        remaining = budget - len(labeled)
        new_points: list[int] = []
        bucket_best: dict[tuple[int, ...], tuple[float, int]] = {}
        for j, ul_idx in enumerate(unlabeled):
            bk = tuple(codes[ul_idx].tolist())
            if bk not in bucket_best or entropy[j] > bucket_best[bk][0]:
                bucket_best[bk] = (entropy[j], int(ul_idx))

        # Sort buckets by their best uncertainty (descending)
        sorted_picks = sorted(bucket_best.values(), key=lambda x: -x[0])
        for _, pt_idx in sorted_picks:
            if len(new_points) >= remaining:
                break
            new_points.append(pt_idx)

        if not new_points:
            break
        labeled.update(new_points)

    labeled_arr = np.array(sorted(labeled), dtype=np.int64)[:budget]
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, labeled_arr)


def strategy_coreset(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """Greedy k-center CoreSet selection in embedding space."""
    n = train_emb.shape[0]
    if budget >= n:
        return _train_and_eval(
            train_emb, train_labels, test_emb, test_labels, np.arange(n)
        )

    # Start with a random point
    selected = [int(rng.randint(0, n))]
    # Distance from each point to its nearest selected center
    min_dist = np.linalg.norm(train_emb - train_emb[selected[0]], axis=1)

    for _ in range(budget - 1):
        # Pick the point farthest from any selected center
        new_idx = int(np.argmax(min_dist))
        selected.append(new_idx)
        # Update distances
        new_dist = np.linalg.norm(train_emb - train_emb[new_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    selected_arr = np.array(selected, dtype=np.int64)
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, selected_arr)


def strategy_badge(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    **kwargs: object,
) -> float:
    """BADGE: gradient embeddings (prob error x features) + k-means++ diversity.

    Ash et al., 2020 — "Deep Batch Active Learning by Diverse, Uncertain
    Gradient Lower Bounds".
    """
    n = train_emb.shape[0]
    n_classes = len(np.unique(train_labels))

    # Start with a small random seed
    seed_size = min(max(10, n_classes * 2), budget)
    labeled = set(rng.choice(n, seed_size, replace=False).tolist())

    while len(labeled) < budget:
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)
        X_labeled = train_emb[labeled_arr]
        y_labeled = train_labels[labeled_arr]

        if len(np.unique(y_labeled)) < 2:
            unlabeled = list(set(range(n)) - labeled)
            add_n = min(1, budget - len(labeled), len(unlabeled))
            labeled.update(rng.choice(unlabeled, add_n, replace=False).tolist())
            continue

        clf = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
        clf.fit(X_labeled, y_labeled)

        unlabeled = np.array(sorted(set(range(n)) - labeled), dtype=np.int64)
        if len(unlabeled) == 0:
            break

        # Compute gradient embeddings: (1 - p_predicted) * x
        probs = clf.predict_proba(train_emb[unlabeled])
        preds = np.argmax(probs, axis=1)
        p_pred = probs[np.arange(len(preds)), preds]
        grad_scale = (1.0 - p_pred).reshape(-1, 1)
        grad_emb = grad_scale * train_emb[unlabeled]  # (n_unlabeled, d)

        # k-means++ selection on gradient embeddings
        remaining = budget - len(labeled)
        batch_size_sel = min(remaining, len(unlabeled))
        new_indices = _kmeans_pp_init(grad_emb, batch_size_sel, rng)
        labeled.update(unlabeled[new_indices].tolist())

    labeled_arr = np.array(sorted(labeled), dtype=np.int64)[:budget]
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, labeled_arr)


def _kmeans_pp_init(
    X: np.ndarray, k: int, rng: np.random.RandomState
) -> np.ndarray:
    """k-means++ initialization: select k diverse points from X."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    # First center: random
    centers_idx = [int(rng.randint(0, n))]
    min_dist_sq = np.sum((X - X[centers_idx[0]]) ** 2, axis=1)

    for _ in range(k - 1):
        # Sample proportional to squared distance
        probs = min_dist_sq / (min_dist_sq.sum() + 1e-30)
        new_idx = int(rng.choice(n, p=probs))
        centers_idx.append(new_idx)
        new_dist_sq = np.sum((X - X[new_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)

    return np.array(centers_idx, dtype=np.int64)


def strategy_typiclust(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    n_clusters: int | None = None,
    **kwargs: object,
) -> float:
    """TypiClust: KMeans + pick point closest to each centroid.

    Hacohen et al., 2022 — simple, strong cold-start baseline.
    """
    if n_clusters is None:
        n_clusters = budget
    n_clusters = min(n_clusters, train_emb.shape[0])

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(rng.randint(0, 2**31)),
        n_init=3,
        batch_size=1024,
    )
    km.fit(train_emb)

    # Pick the point closest to each centroid
    selected: list[int] = []
    for c in range(n_clusters):
        if len(selected) >= budget:
            break
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(train_emb[members] - km.cluster_centers_[c], axis=1)
        closest = members[np.argmin(dists)]
        selected.append(int(closest))

    selected_arr = np.array(selected[:budget], dtype=np.int64)
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, selected_arr)


def strategy_probcover(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    delta: float | None = None,
    **kwargs: object,
) -> float:
    """ProbCover: greedy set cover with kNN ball graph.

    Yehuda et al., 2022 — pick points that cover the most uncovered neighbors.
    Delta auto-tuned as median pairwise distance / sqrt(d).
    """
    n = train_emb.shape[0]
    d = train_emb.shape[1]

    if delta is None:
        # Auto-tune delta: subsample for speed, then median / sqrt(d)
        subsample_n = min(2000, n)
        idx_sub = rng.choice(n, subsample_n, replace=False)
        dists_sub = pairwise_distances(train_emb[idx_sub], metric="euclidean")
        triu_idx = np.triu_indices(subsample_n, k=1)
        median_dist = float(np.median(dists_sub[triu_idx]))
        delta = median_dist / math.sqrt(max(d, 1))

    # Greedy set cover
    covered = np.zeros(n, dtype=bool)
    selected: list[int] = []
    chunk_size = 2000

    for _ in range(budget):
        if covered.all():
            break

        best_idx = -1
        best_cover = -1

        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            dists_chunk = pairwise_distances(
                train_emb[i:end_i], train_emb, metric="euclidean"
            )
            within = dists_chunk <= delta
            uncovered_covered = within & (~covered[np.newaxis, :])
            cover_counts = uncovered_covered.sum(axis=1)

            local_best = int(np.argmax(cover_counts))
            if cover_counts[local_best] > best_cover:
                best_cover = int(cover_counts[local_best])
                best_idx = i + local_best

        if best_idx < 0 or best_cover == 0:
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            best_idx = int(rng.choice(uncovered_idx))

        selected.append(best_idx)
        dists_to_sel = np.linalg.norm(train_emb - train_emb[best_idx], axis=1)
        covered[dists_to_sel <= delta] = True

    selected_arr = np.array(selected[:budget], dtype=np.int64)
    return _train_and_eval(train_emb, train_labels, test_emb, test_labels, selected_arr)


# =============================================================================
# RQ one-hot encoding helpers
# =============================================================================


def build_rq_onehot(codes: np.ndarray, nbits: int) -> np.ndarray:
    """Build one-hot encoding of RQ trunk codes.

    For each point, codes are (c_0, c_1, ..., c_{d_cut-1}) with c_l in
    {0, ..., 2^nbits - 1}. One-hot: for each level l, binary vector of
    length 2^nbits with a 1 at position c_l. Concatenate across levels.

    Args:
        codes: Trunk codes, shape (n, d_cut).
        nbits: Bits per RQ level.

    Returns:
        One-hot matrix, shape (n, d_cut * 2^nbits).
    """
    n = codes.shape[0]
    d_cut = codes.shape[1]
    k = 1 << nbits  # 2^nbits

    onehot = np.zeros((n, d_cut * k), dtype=np.float32)
    for level in range(d_cut):
        col_offset = level * k
        for i in range(n):
            onehot[i, col_offset + int(codes[i, level])] = 1.0
    return onehot


def build_rq_augmented_features(
    emb: np.ndarray, codes: np.ndarray, nbits: int
) -> np.ndarray:
    """Build RQ-augmented feature matrix: [x; gamma * rq_onehot].

    gamma = sqrt(d / d_cut) balances continuous vs discrete feature norms.

    Args:
        emb: Embedding features, shape (n, d).
        codes: Trunk codes, shape (n, d_cut).
        nbits: Bits per RQ level.

    Returns:
        Augmented features, shape (n, d + d_cut * 2^nbits).
    """
    d = emb.shape[1]
    d_cut = codes.shape[1]
    gamma = math.sqrt(d / max(d_cut, 1))

    rq_oh = build_rq_onehot(codes, nbits)
    return np.hstack([emb, gamma * rq_oh])


def strategy_rqaug_uncertainty(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    train_codes: np.ndarray | None = None,
    test_codes: np.ndarray | None = None,
    nbits: int = 3,
    **kwargs: object,
) -> float:
    """RQ-augmented uncertainty sampling: uncertainty on [x; gamma*rq_onehot]."""
    if train_codes is None or test_codes is None:
        return strategy_uncertainty(
            budget, train_emb, train_labels, test_emb, test_labels, rng
        )

    train_aug = build_rq_augmented_features(train_emb, train_codes, nbits)
    test_aug = build_rq_augmented_features(test_emb, test_codes, nbits)
    return strategy_uncertainty(
        budget, train_aug, train_labels, test_aug, test_labels, rng
    )


def strategy_rqaug_badge(
    budget: int,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    rng: np.random.RandomState,
    train_codes: np.ndarray | None = None,
    test_codes: np.ndarray | None = None,
    nbits: int = 3,
    **kwargs: object,
) -> float:
    """RQ-augmented BADGE: BADGE on [x; gamma*rq_onehot] features."""
    if train_codes is None or test_codes is None:
        return strategy_badge(
            budget, train_emb, train_labels, test_emb, test_labels, rng
        )

    train_aug = build_rq_augmented_features(train_emb, train_codes, nbits)
    test_aug = build_rq_augmented_features(test_emb, test_codes, nbits)
    return strategy_badge(
        budget, train_aug, train_labels, test_aug, test_labels, rng
    )


# =============================================================================
# Main experiment
# =============================================================================


def run_label_efficiency(config: LabelEfficiencyConfig) -> dict[str, object]:
    """End-to-end label efficiency experiment."""
    import faiss

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Auto-dcut: override nbits/d_cut from per-dataset configs
    if config.auto_dcut:
        from scripts.experiments.run_real_nts_experiment import (
            DATASET_FEATURES_RQ_CONFIGS,
            DATASET_RQ_CONFIGS,
        )

        rq_configs = (
            DATASET_FEATURES_RQ_CONFIGS
            if config.feature_mode in ("raw", "selected", "lgbm", "iforest")
            else DATASET_RQ_CONFIGS
        )
        if config.dataset in rq_configs:
            auto_nbits, auto_dcut = rq_configs[config.dataset]
            config.nbits = auto_nbits
            config.d_cut_values = [auto_dcut]
            logger.info(
                f"auto_dcut: overriding nbits={auto_nbits}, d_cut_values=[{auto_dcut}] "
                f"for dataset={config.dataset} (feature_mode={config.feature_mode})"
            )

    logger.info(f"Loading dataset: {config.dataset}")
    train_dataset, _val_dataset, test_dataset, metadata = load_dataset(config.dataset)

    n_classes = metadata.n_classes
    logger.info(
        f"Loaded: {metadata.n_samples} samples, "
        f"{metadata.n_continuous} cont, {metadata.n_categorical} cat, "
        f"{n_classes} classes"
    )

    # Resolve auto_dcut: use per-dataset diagnostic (nbits, d_cut)
    if config.auto_dcut:
        from scripts.experiments.run_real_nts_experiment import (
            DATASET_FEATURES_RQ_CONFIGS,
            DATASET_IFOREST_RQ_CONFIGS,
            DATASET_LGBM_RQ_CONFIGS,
            DATASET_RQ_CONFIGS,
        )

        if config.embedding_type == "lgbm":
            rq_configs = DATASET_LGBM_RQ_CONFIGS
        elif config.embedding_type == "iforest":
            rq_configs = DATASET_IFOREST_RQ_CONFIGS
        elif config.feature_mode in ("raw", "selected"):
            rq_configs = DATASET_FEATURES_RQ_CONFIGS
        else:
            rq_configs = DATASET_RQ_CONFIGS
        if config.dataset in rq_configs:
            diag_nbits, diag_dcut = rq_configs[config.dataset]
            config.nbits = diag_nbits
            config.d_cut_values = [diag_dcut]
            logger.info(
                f"  auto_dcut: using diagnostic nbits={diag_nbits}, "
                f"d_cut={diag_dcut} for {config.dataset}"
            )
        else:
            logger.warning(
                f"  auto_dcut: no diagnostic for {config.dataset}, "
                f"falling back to d_cut_values={config.d_cut_values}"
            )

    per_seed_results: list[dict[str, object]] = []

    # Determine iteration count: raw/selected features are deterministic
    n_feature_iters = (
        1 if config.feature_mode in ("raw", "selected") else config.n_encoder_seeds
    )

    for enc_idx in range(n_feature_iters):
        encoder_seed = config.seed + enc_idx * 10000
        logger.info(
            f"\n{'=' * 60}\n  {'Feature' if config.feature_mode != 'embeddings' else 'Encoder'} "
            f"seed {enc_idx + 1}/{n_feature_iters} "
            f"(seed={encoder_seed})\n{'=' * 60}"
        )

        np.random.seed(encoder_seed)
        torch.manual_seed(encoder_seed)

        if config.feature_mode in ("raw", "selected"):
            # Use raw/selected features instead of encoder embeddings
            cache = FeatureCache(config.feature_cache_dir)
            train_features_np, train_labels_np_raw = cache.get_or_compute_raw_features(
                config.dataset,
                lambda: collect_raw_features(train_dataset),
            )
            test_features_np, test_labels_np_raw = collect_raw_features(test_dataset)

            if config.feature_mode == "selected":
                selected_indices, mi_scores = cache.get_or_compute_selection(
                    config.dataset,
                    "supervised",
                    lambda: select_features_supervised(
                        train_features_np, train_labels_np_raw
                    ),
                )
                train_features_np = train_features_np[:, selected_indices]
                test_features_np = test_features_np[:, selected_indices]
                logger.info(
                    f"  Feature selection: {len(selected_indices)} of "
                    f"{mi_scores.shape[0]} features selected (supervised MI)"
                )

            # Normalize before RQ
            train_emb_np, test_emb_np, _ = normalize_for_rq(
                train_features_np, test_features_np
            )
            train_labels_np = train_labels_np_raw
            test_labels_np = test_labels_np_raw
            logger.info(
                f"  {config.feature_mode} features: train={train_emb_np.shape}, "
                f"test={test_emb_np.shape}"
            )
        elif config.feature_mode in ("lgbm", "iforest"):
            # Use lgbm/iforest embeddings via EmbeddingExtractor
            cache = FeatureCache(config.feature_cache_dir)
            train_features_np, train_labels_np_raw = cache.get_or_compute_raw_features(
                config.dataset,
                lambda: collect_raw_features(train_dataset),
            )
            test_features_np, test_labels_np_raw = collect_raw_features(test_dataset)

            extractor = EmbeddingExtractor(
                EmbeddingConfig(
                    embedding_type=config.feature_mode,
                    random_state=encoder_seed,
                )
            )
            train_emb_raw = extractor.fit_transform(
                train_features_np,
                y_train=train_labels_np_raw if config.feature_mode == "lgbm" else None,
            )
            test_emb_raw = extractor.transform(test_features_np)

            train_emb_np, test_emb_np, _ = normalize_for_rq(train_emb_raw, test_emb_raw)
            train_labels_np = train_labels_np_raw
            test_labels_np = test_labels_np_raw
            logger.info(
                f"  {config.feature_mode} features: train={train_emb_np.shape}, "
                f"test={test_emb_np.shape}"
            )
        else:
            train_loader: DataLoader = DataLoader(  # pyre-ignore[11]
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(encoder_seed),
            )
            test_loader: DataLoader = DataLoader(  # pyre-ignore[11]
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )

            # Train encoder
            logger.info("  Pre-training encoder...")
            encoder = MLPEncoder(
                n_categories=metadata.category_sizes,
                n_continuous=metadata.n_continuous,
                dim=config.embedding_dim,
                hidden_dims=[config.hidden_dim, config.hidden_dim],
            ).to(device)
            encoder = pretrain_encoder(
                encoder,
                train_loader,
                n_classes,
                config.embedding_dim,
                config.lr,
                config.weight_decay,
                config.encoder_epochs,
                device,
            )

            # Collect embeddings
            logger.info("  Collecting embeddings...")
            train_emb, train_labels = collect_embeddings_and_labels(
                encoder, train_loader, device
            )
            test_emb, test_labels = collect_embeddings_and_labels(
                encoder, test_loader, device
            )

            train_emb_np_raw = train_emb.detach().numpy().astype(np.float32)
            test_emb_np_raw = test_emb.detach().numpy().astype(np.float32)

            # Standardize embeddings before RQ (critical for codebook quality)
            train_emb_np, test_emb_np, _ = normalize_for_rq(
                train_emb_np_raw, test_emb_np_raw
            )

            train_labels_np = train_labels.numpy()
            test_labels_np = test_labels.numpy()

        n_train = train_emb_np.shape[0]
        dim = train_emb_np.shape[1]

        # Full-data baseline
        full_acc = _train_and_eval(
            train_emb_np,
            train_labels_np,
            test_emb_np,
            test_labels_np,
            np.arange(n_train),
        )
        logger.info(f"  Full-data baseline accuracy: {full_acc:.4f}")
        print(
            f"  [{config.dataset}] Encoder seed {enc_idx + 1}/{config.n_encoder_seeds}, full-data baseline: {full_acc:.4f}",
            flush=True,
        )

        # Fit RQ at each d_cut
        rq_codes_train: dict[int, np.ndarray] = {}
        rq_codes_test: dict[int, np.ndarray] = {}
        rq_n_unique: dict[int, int] = {}
        rq_h_y_z: dict[int, float] = {}

        for d_cut in config.d_cut_values:
            rq = faiss.ResidualQuantizer(dim, d_cut, config.nbits)
            rq.train_type = faiss.ResidualQuantizer.Train_default
            rq.verbose = False
            rq.train(train_emb_np)

            codes_tr = rq.compute_codes(train_emb_np).astype(np.int64)
            codes_te = rq.compute_codes(test_emb_np).astype(np.int64)
            rq_codes_train[d_cut] = codes_tr
            rq_codes_test[d_cut] = codes_te

            n_unique = len(set(tuple(c) for c in codes_tr))
            rq_n_unique[d_cut] = n_unique
            h_yz = conditional_entropy(train_labels_np, codes_tr)
            rq_h_y_z[d_cut] = h_yz
            logger.info(
                f"  RQ d_cut={d_cut}: {n_unique} unique codes, H(Y|Z)={h_yz:.4f}"
            )

        # Fit KMeans at matched cluster counts
        km_labels_train: dict[int, np.ndarray] = {}
        km_labels_test: dict[int, np.ndarray] = {}

        for d_cut in config.d_cut_values:
            n_clusters = min(rq_n_unique[d_cut], n_train)
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=encoder_seed,
                n_init=3,
                batch_size=1024,
            )
            km.fit(train_emb_np)
            km_labels_train[d_cut] = km.labels_
            km_labels_test[d_cut] = km.predict(test_emb_np)

            km_h_yz = conditional_entropy(train_labels_np, km.labels_.reshape(-1, 1))
            logger.info(f"  KM k={n_clusters}: H(Y|Z)={km_h_yz:.4f}")

        # ----- Evaluate strategies at each budget -----
        # Methods: base + per-d_cut variants
        method_names: list[str] = [
            "Random",
            "Uncertainty",
            "CoreSet",
            "BADGE",
            "TypiClust",
            "ProbCover",
        ]
        for d_cut in config.d_cut_values:
            method_names.extend(
                [
                    f"RQ-Strat-d{d_cut}",
                    f"KM-Strat-d{d_cut}",
                    f"RQ-Prop-d{d_cut}",
                    f"KM-Prop-d{d_cut}",
                    f"RQ-SeedUnc-d{d_cut}",
                    f"RQ-DivUnc-d{d_cut}",
                    f"RQAug-Unc-d{d_cut}",
                    f"RQAug-BADGE-d{d_cut}",
                ]
            )

        # results[method][budget] = list of accuracies across label seeds
        budget_results: dict[str, dict[int, list[float]]] = {
            m: {b: [] for b in config.label_budgets} for m in method_names
        }

        for ls_idx in range(config.n_label_seeds):
            label_seed = encoder_seed + ls_idx * 100 + 1
            logger.info(
                f"\n  Label seed {ls_idx + 1}/{config.n_label_seeds} (seed={label_seed})"
            )

            for budget in config.label_budgets:
                rng = np.random.RandomState(label_seed + budget)

                # Random
                acc = strategy_random(
                    budget,
                    train_emb_np,
                    train_labels_np,
                    test_emb_np,
                    test_labels_np,
                    rng,
                )
                budget_results["Random"][budget].append(acc)

                # Uncertainty
                rng_u = np.random.RandomState(label_seed + budget + 10000)
                acc = strategy_uncertainty(
                    budget,
                    train_emb_np,
                    train_labels_np,
                    test_emb_np,
                    test_labels_np,
                    rng_u,
                )
                budget_results["Uncertainty"][budget].append(acc)

                # CoreSet
                rng_cs = np.random.RandomState(label_seed + budget + 20000)
                acc = strategy_coreset(
                    budget,
                    train_emb_np,
                    train_labels_np,
                    test_emb_np,
                    test_labels_np,
                    rng_cs,
                )
                budget_results["CoreSet"][budget].append(acc)

                # BADGE
                rng_bg = np.random.RandomState(label_seed + budget + 30000)
                acc = strategy_badge(
                    budget,
                    train_emb_np,
                    train_labels_np,
                    test_emb_np,
                    test_labels_np,
                    rng_bg,
                )
                budget_results["BADGE"][budget].append(acc)

                # TypiClust
                rng_tc = np.random.RandomState(label_seed + budget + 40000)
                acc = strategy_typiclust(
                    budget,
                    train_emb_np,
                    train_labels_np,
                    test_emb_np,
                    test_labels_np,
                    rng_tc,
                )
                budget_results["TypiClust"][budget].append(acc)

                # ProbCover
                rng_pc = np.random.RandomState(label_seed + budget + 50000)
                acc = strategy_probcover(
                    budget,
                    train_emb_np,
                    train_labels_np,
                    test_emb_np,
                    test_labels_np,
                    rng_pc,
                )
                budget_results["ProbCover"][budget].append(acc)

                for d_cut in config.d_cut_values:
                    # RQ-Stratified
                    rng_rq = np.random.RandomState(label_seed + budget + d_cut * 1000)
                    acc = strategy_stratified(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        rq_codes_train[d_cut],
                        rng_rq,
                    )
                    budget_results[f"RQ-Strat-d{d_cut}"][budget].append(acc)

                    # KM-Stratified
                    rng_km = np.random.RandomState(label_seed + budget + d_cut * 2000)
                    acc = strategy_stratified(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        km_labels_train[d_cut].reshape(-1, 1),
                        rng_km,
                    )
                    budget_results[f"KM-Strat-d{d_cut}"][budget].append(acc)

                    # RQ-Propagate
                    rng_rqp = np.random.RandomState(label_seed + budget + d_cut * 3000)
                    acc = strategy_propagate(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        rq_codes_train[d_cut],
                        rq_codes_test[d_cut],
                        n_classes,
                        rng_rqp,
                    )
                    budget_results[f"RQ-Prop-d{d_cut}"][budget].append(acc)

                    # KM-Propagate
                    rng_kmp = np.random.RandomState(label_seed + budget + d_cut * 4000)
                    acc = strategy_propagate(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        km_labels_train[d_cut].reshape(-1, 1),
                        km_labels_test[d_cut].reshape(-1, 1),
                        n_classes,
                        rng_kmp,
                    )
                    budget_results[f"KM-Prop-d{d_cut}"][budget].append(acc)

                    # RQ-Seeded Uncertainty
                    rng_rsu = np.random.RandomState(label_seed + budget + d_cut * 5000)
                    acc = strategy_rq_seeded_uncertainty(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        rq_codes_train[d_cut],
                        rng_rsu,
                    )
                    budget_results[f"RQ-SeedUnc-d{d_cut}"][budget].append(acc)

                    # RQ-Diverse Uncertainty
                    rng_rdu = np.random.RandomState(label_seed + budget + d_cut * 6000)
                    acc = strategy_rq_diverse_uncertainty(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        rq_codes_train[d_cut],
                        rng_rdu,
                    )
                    budget_results[f"RQ-DivUnc-d{d_cut}"][budget].append(acc)

                    # RQ-Augmented Uncertainty
                    rng_rau = np.random.RandomState(label_seed + budget + d_cut * 7000)
                    acc = strategy_rqaug_uncertainty(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        rng_rau,
                        train_codes=rq_codes_train[d_cut],
                        test_codes=rq_codes_test[d_cut],
                        nbits=config.nbits,
                    )
                    budget_results[f"RQAug-Unc-d{d_cut}"][budget].append(acc)

                    # RQ-Augmented BADGE
                    rng_rab = np.random.RandomState(label_seed + budget + d_cut * 8000)
                    acc = strategy_rqaug_badge(
                        budget,
                        train_emb_np,
                        train_labels_np,
                        test_emb_np,
                        test_labels_np,
                        rng_rab,
                        train_codes=rq_codes_train[d_cut],
                        test_codes=rq_codes_test[d_cut],
                        nbits=config.nbits,
                    )
                    budget_results[f"RQAug-BADGE-d{d_cut}"][budget].append(acc)

        # Log learning curves
        logger.info(
            f"\n  Learning curves (test acc, mean over {config.n_label_seeds} seeds):"
        )
        header = f"  {'Budget':>6s}" + "".join(f"  {m:>14s}" for m in method_names)
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))

        for budget in config.label_budgets:
            row = f"  {budget:>6d}"
            for m in method_names:
                vals = budget_results[m][budget]
                mean = np.mean(vals)
                row += f"  {mean:>14.4f}"
            logger.info(row)

        logger.info(f"\n  Full-data baseline: {full_acc:.4f}")

        # Compute efficiency: labels needed to reach 90%, 95%, 99% of full-data acc
        logger.info(
            f"\n  Labels needed to reach X% of full-data accuracy ({full_acc:.4f}):"
        )
        for threshold_pct in [90, 95, 99]:
            threshold = full_acc * threshold_pct / 100
            row = f"    {threshold_pct}% ({threshold:.4f}):"
            for m in method_names:
                found = False
                for budget in config.label_budgets:
                    mean = np.mean(budget_results[m][budget])
                    if mean >= threshold:
                        row += f"  {m}={budget}"
                        found = True
                        break
                if not found:
                    row += f"  {m}=>{config.label_budgets[-1]}"
            logger.info(row)

        # Store seed results
        seed_result: dict[str, object] = {
            "encoder_seed": encoder_seed,
            "n_train": n_train,
            "n_test": int(test_emb_np.shape[0]),
            "full_data_acc": full_acc,
            "rq_n_unique": rq_n_unique,
            "rq_h_y_given_z": rq_h_y_z,
            "methods": {},
        }
        for m in method_names:
            seed_result["methods"][m] = {  # pyre-ignore[16]
                str(b): {
                    "mean": float(np.mean(budget_results[m][b])),
                    "std": float(np.std(budget_results[m][b])),
                    "values": budget_results[m][b],
                }
                for b in config.label_budgets
            }
        per_seed_results.append(seed_result)

    # ----- Aggregate across encoder seeds -----
    logger.info("\n" + "=" * 80)
    logger.info(
        f"AGGREGATE: {config.dataset} ({config.n_encoder_seeds} encoder seeds "
        f"x {config.n_label_seeds} label seeds)"
    )
    logger.info("=" * 80)

    method_names_all = list(per_seed_results[0]["methods"].keys())
    full_accs = [s["full_data_acc"] for s in per_seed_results]
    full_acc_mean = float(np.mean(full_accs))

    logger.info(f"Full-data baseline: {full_acc_mean:.4f} +/- {np.std(full_accs):.4f}")

    aggregate: dict[str, dict[str, dict[str, float]]] = {}

    logger.info(f"\nLearning curves (test acc, mean +/- std):")
    header = f"{'Budget':>6s}" + "".join(f"  {m:>14s}" for m in method_names_all)
    logger.info(header)
    logger.info("-" * len(header))

    for budget in config.label_budgets:
        row = f"{budget:>6d}"
        for m in method_names_all:
            # Collect all values across all seeds
            all_vals: list[float] = []
            for s in per_seed_results:
                all_vals.extend(s["methods"][m][str(budget)]["values"])
            mean = float(np.mean(all_vals))
            std = float(np.std(all_vals))
            if m not in aggregate:
                aggregate[m] = {}
            aggregate[m][str(budget)] = {"mean": mean, "std": std}
            row += f"  {mean:>7.4f}±{std:.3f}"
        logger.info(row)

    # Efficiency table
    logger.info(f"\nLabels to reach X% of full-data acc ({full_acc_mean:.4f}):")
    for threshold_pct in [90, 95, 99]:
        threshold = full_acc_mean * threshold_pct / 100
        logger.info(f"  {threshold_pct}% ({threshold:.4f}):")
        for m in method_names_all:
            for budget in config.label_budgets:
                if aggregate[m][str(budget)]["mean"] >= threshold:
                    logger.info(f"    {m:<20s}: {budget} labels")
                    break
            else:
                logger.info(f"    {m:<20s}: >{config.label_budgets[-1]} labels")

    # ----- Per-budget significance tests -----
    significance: dict[str, dict[str, object]] = {}
    if config.n_encoder_seeds >= 2:
        logger.info("\n" + "=" * 80)
        logger.info("SIGNIFICANCE TESTS (best-ours vs best-baseline, per budget)")
        logger.info("=" * 80)
        for budget in config.label_budgets:
            method_seed_means: dict[str, list[float]] = {}
            for m in method_names_all:
                seed_means: list[float] = []
                for s in per_seed_results:
                    vals = s["methods"][m][str(budget)]["values"]
                    seed_means.append(float(np.mean(vals)))
                method_seed_means[m] = seed_means

            ours = {m for m in method_names_all if "RQ" in m}
            sig = compute_pairwise_significance(
                method_seed_means,
                ours_methods=ours,
                higher_is_better=True,
            )
            if sig:
                log_significance(sig, logger, metric_name=f"acc@{budget}")
                significance[str(budget)] = sig

    result: dict[str, object] = {
        "dataset": config.dataset,
        "n_classes": n_classes,
        "config": asdict(config),
        "per_seed_results": per_seed_results,
        "aggregate": aggregate,
        "full_data_acc_mean": full_acc_mean,
        "full_data_acc_std": float(np.std(full_accs)),
        "significance": significance,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label efficiency: trunk codes as labeling oracle"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=list(DATASET_REGISTRY.keys() - {"synthetic"}),
        help="Dataset to use",
    )
    parser.add_argument(
        "--n-encoder-seeds",
        type=int,
        default=3,
        help="Number of encoder random seeds",
    )
    parser.add_argument(
        "--n-label-seeds",
        type=int,
        default=5,
        help="Number of label selection random seeds",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=6,
        help="Bits per RQ level",
    )
    parser.add_argument(
        "--d-cut-values",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="d_cut values to evaluate",
    )
    parser.add_argument(
        "--label-budgets",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        help="Number of labeled points to try",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Encoder embedding dimension",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/label_efficiency_results",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="embeddings",
        choices=["embeddings", "raw", "selected", "lgbm", "iforest"],
        help="Input space for RQ: encoder embeddings, raw features, selected features, "
        "lgbm prob embeddings, or iforest path embeddings",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="results/feature_cache",
        help="Directory for caching raw features and feature selection results",
    )
    parser.add_argument(
        "--auto-dcut",
        action="store_true",
        default=True,
        help="Use per-dataset diagnostic (nbits, d_cut) instead of sweeping d_cut_values",
    )
    parser.add_argument(
        "--no-auto-dcut",
        action="store_false",
        dest="auto_dcut",
        help="Disable auto_dcut and sweep d_cut_values",
    )
    args = parser.parse_args()

    config = LabelEfficiencyConfig(
        dataset=args.dataset,
        n_encoder_seeds=args.n_encoder_seeds,
        n_label_seeds=args.n_label_seeds,
        nbits=args.nbits,
        d_cut_values=args.d_cut_values,
        auto_dcut=args.auto_dcut,
        label_budgets=args.label_budgets,
        embedding_dim=args.embedding_dim,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
        feature_mode=args.feature_mode,
        feature_cache_dir=args.feature_cache_dir,
    )

    logger.info("=" * 60)
    logger.info("LABEL EFFICIENCY EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset}, feature_mode: {config.feature_mode}")
    logger.info(
        f"n_encoder_seeds: {config.n_encoder_seeds}, "
        f"n_label_seeds: {config.n_label_seeds}"
    )
    logger.info(f"d_cut_values: {config.d_cut_values}, nbits: {config.nbits}")
    logger.info(f"label_budgets: {config.label_budgets}")

    start_time = time.time()
    results = run_label_efficiency(config)
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir / f"label_efficiency_{config.dataset}_nbits{config.nbits}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
