#!/usr/bin/env python3


"""Retrieval / few-shot classification experiment on real tabular datasets.

Demonstrates that RQ trunk codes serve as label-aware hash buckets for
retrieval and few-shot classification. Trunk codes hash points into
neighborhoods that are label-pure, so k-NN in trunk-code space yields
better classification than brute-force k-NN or KMeans hashing.

Evaluation methods (6 total):
    1. BruteForce-kNN: Exact k-NN on raw embeddings (L2 distance)
    2. RQ-Hash-kNN: k-NN restricted to same trunk code bucket
    3. KM-Hash-kNN: k-NN restricted to same KMeans cluster
    4. Faiss-IVF: Faiss IVF approximate nearest neighbor search
    5. RQ-Denoised-kNN: k-NN on trunk reconstructions (decoded trunk vectors)
    6. KM-Denoised-kNN: k-NN on KMeans centroid reconstructions

Metrics: Precision@k, Recall@k, MRR, Few-shot accuracy.

Example:
    python3 scripts/experiments/run_retrieval_experiment.py -- \\
        --dataset adult --n-encoder-seeds 1 --device cuda
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
from modules.encoders.tab_transformer import MLPEncoder
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Duplicated helpers (from run_real_nts_experiment.py / run_dcut_diagnostic.py)
# Each experiment script is self-contained (python_binary, not python_library).
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


def compute_trunk_codes(
    embeddings: Tensor,
    d_cut: int,
    nbits: int,
) -> tuple[Tensor, object]:
    """Fit a trunk-only RQ and extract codes from embeddings.

    Returns:
        codes: int64 tensor of shape (N, d_cut)
        rq: the trained faiss.ResidualQuantizer
    """
    import faiss

    embeddings_np = embeddings.detach().numpy().astype(np.float32)
    dim = embeddings_np.shape[1]

    rq = faiss.ResidualQuantizer(dim, d_cut, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = False
    rq.train(embeddings_np)

    codes_np = rq.compute_codes(embeddings_np)
    codes = torch.from_numpy(codes_np.astype(np.int64))
    return codes, rq


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
class RetrievalConfig:
    """Configuration for retrieval / few-shot classification experiment."""

    dataset: str = "adult"
    n_encoder_seeds: int = 3
    seed: int = 42

    # Encoder
    embedding_dim: int = 64
    hidden_dim: int = 128
    encoder_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256

    # RQ
    n_levels: int = 8
    nbits: int = 6
    d_cut: int = 2

    # KMeans ablation
    km_clusters: int | None = None  # default: 2^nbits

    # Retrieval
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    n_shot_values: list[int] = field(default_factory=lambda: [1, 5, 10])

    device: str = "cuda"


# =============================================================================
# Evaluation helpers
# =============================================================================

METHOD_NAMES: list[str] = [
    "BruteForce-kNN",
    "RQ-Hash-kNN",
    "KM-Hash-kNN",
    "Faiss-IVF",
    "RQ-Denoised-kNN",
    "KM-Denoised-kNN",
]


def _build_hash_index(
    codes: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Build a mapping from code tuple -> array of train indices."""
    bucket: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(codes.shape[0]):
        key = tuple(codes[i].tolist())
        bucket[key].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in bucket.items()}


def _knn_from_dists(
    dists: np.ndarray,
    train_labels: np.ndarray,
    query_label: int,
    k: int,
    n_same_class_in_db: int,
) -> tuple[float, float, float]:
    """Compute precision@k, recall@k, and reciprocal rank from distance array.

    Args:
        dists: (N_candidates,) L2 distances from query to candidates.
        train_labels: (N_candidates,) labels of the candidates.
        query_label: label of the query point.
        k: number of neighbors.
        n_same_class_in_db: total number of same-class items in the database.

    Returns:
        (precision_at_k, recall_at_k, reciprocal_rank)
    """
    actual_k = min(k, len(dists))
    if actual_k == 0:
        return 0.0, 0.0, 0.0

    if actual_k >= len(dists):
        # All candidates fit in top-k, just sort by distance
        topk_idx = np.argsort(dists)
    else:
        topk_idx = np.argpartition(dists, actual_k)[:actual_k]
        # Sort the top-k by distance for MRR computation
        sorted_order = np.argsort(dists[topk_idx])
        topk_idx = topk_idx[sorted_order]
    topk_labels = train_labels[topk_idx]

    matches = (topk_labels == query_label).astype(np.float64)
    precision = float(matches.sum() / actual_k)

    recall = (
        float(matches.sum() / n_same_class_in_db) if n_same_class_in_db > 0 else 0.0
    )

    # Reciprocal rank: 1/rank of first same-class neighbor
    same_class_positions = np.where(matches > 0)[0]
    rr = (
        float(1.0 / (same_class_positions[0] + 1))
        if len(same_class_positions) > 0
        else 0.0
    )

    return precision, recall, rr


def evaluate_retrieval(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    trunk_codes_train: np.ndarray,
    trunk_codes_test: np.ndarray,
    trunk_recon_train: np.ndarray,
    trunk_recon_test: np.ndarray,
    km_labels_train: np.ndarray,
    km_labels_test: np.ndarray,
    km_recon_train: np.ndarray,
    km_recon_test: np.ndarray,
    faiss_ivf: object,
    k_values: list[int],
) -> dict[str, dict[str, list[float] | float]]:
    """Evaluate all 6 retrieval methods.

    Returns:
        method_name -> {
            "precision_at_k": [p@k1, p@k2, ...],
            "recall_at_k": [r@k1, r@k2, ...],
            "mrr": float,
        }
    """
    import faiss as faiss_lib

    n_test = test_emb.shape[0]
    max_k = max(k_values)

    # Precompute class counts in training set
    class_counts_train: dict[int, int] = {}
    for lab in train_labels:
        class_counts_train[lab] = class_counts_train.get(lab, 0) + 1

    # Build hash indices
    rq_hash = _build_hash_index(trunk_codes_train)
    km_hash = _build_hash_index(km_labels_train.reshape(-1, 1))

    # Build brute-force Faiss index on raw embeddings
    bf_index = faiss_lib.IndexFlatL2(train_emb.shape[1])
    bf_index.add(train_emb)

    # Build brute-force index on trunk reconstructions
    recon_index = faiss_lib.IndexFlatL2(trunk_recon_train.shape[1])
    recon_index.add(trunk_recon_train)

    # Build brute-force index on KMeans reconstructions
    km_recon_index = faiss_lib.IndexFlatL2(km_recon_train.shape[1])
    km_recon_index.add(km_recon_train)

    # Per-method accumulators
    # method -> k_idx -> list of per-query values
    precision_accum: dict[str, list[list[float]]] = {
        m: [[] for _ in k_values] for m in METHOD_NAMES
    }
    recall_accum: dict[str, list[list[float]]] = {
        m: [[] for _ in k_values] for m in METHOD_NAMES
    }
    mrr_accum: dict[str, list[float]] = {m: [] for m in METHOD_NAMES}

    # Process test queries in batches for brute-force / IVF / denoised methods
    # (these support batch queries via Faiss)
    batch_size = 1024

    for batch_start in range(0, n_test, batch_size):
        batch_end = min(batch_start + batch_size, n_test)
        batch_queries = test_emb[batch_start:batch_end]
        batch_labels = test_labels[batch_start:batch_end]
        b_size = batch_end - batch_start

        # --- BruteForce-kNN ---
        bf_dists, bf_indices = bf_index.search(batch_queries, max_k)
        for i in range(b_size):
            q_label = int(batch_labels[i])
            n_same = class_counts_train.get(q_label, 0)
            neighbor_labels = train_labels[bf_indices[i]]
            for ki, k in enumerate(k_values):
                actual_k = min(k, max_k)
                top_labels = neighbor_labels[:actual_k]
                matches = (top_labels == q_label).astype(np.float64)
                precision_accum["BruteForce-kNN"][ki].append(
                    float(matches.sum() / actual_k)
                )
                recall_accum["BruteForce-kNN"][ki].append(
                    float(matches.sum() / n_same) if n_same > 0 else 0.0
                )
            # MRR: rank of first same-class neighbor
            same_mask = neighbor_labels == q_label
            positions = np.where(same_mask)[0]
            mrr_accum["BruteForce-kNN"].append(
                float(1.0 / (positions[0] + 1)) if len(positions) > 0 else 0.0
            )

        # --- Faiss-IVF ---
        ivf_dists, ivf_indices = faiss_ivf.search(
            batch_queries, max_k
        )  # pyre-ignore[16]
        for i in range(b_size):
            q_label = int(batch_labels[i])
            n_same = class_counts_train.get(q_label, 0)
            # IVF may return -1 for unfound neighbors
            valid = ivf_indices[i] >= 0
            valid_indices = ivf_indices[i][valid]
            neighbor_labels = (
                train_labels[valid_indices]
                if len(valid_indices) > 0
                else np.array([], dtype=train_labels.dtype)
            )
            for ki, k in enumerate(k_values):
                actual_k = min(k, len(neighbor_labels))
                if actual_k == 0:
                    precision_accum["Faiss-IVF"][ki].append(0.0)
                    recall_accum["Faiss-IVF"][ki].append(0.0)
                    continue
                top_labels = neighbor_labels[:actual_k]
                matches = (top_labels == q_label).astype(np.float64)
                precision_accum["Faiss-IVF"][ki].append(float(matches.sum() / actual_k))
                recall_accum["Faiss-IVF"][ki].append(
                    float(matches.sum() / n_same) if n_same > 0 else 0.0
                )
            positions = (
                np.where(neighbor_labels == q_label)[0]
                if len(neighbor_labels) > 0
                else np.array([])
            )
            mrr_accum["Faiss-IVF"].append(
                float(1.0 / (positions[0] + 1)) if len(positions) > 0 else 0.0
            )

        # --- RQ-Denoised-kNN ---
        recon_queries = trunk_recon_test[batch_start:batch_end]
        rd_dists, rd_indices = recon_index.search(recon_queries, max_k)
        for i in range(b_size):
            q_label = int(batch_labels[i])
            n_same = class_counts_train.get(q_label, 0)
            neighbor_labels = train_labels[rd_indices[i]]
            for ki, k in enumerate(k_values):
                actual_k = min(k, max_k)
                top_labels = neighbor_labels[:actual_k]
                matches = (top_labels == q_label).astype(np.float64)
                precision_accum["RQ-Denoised-kNN"][ki].append(
                    float(matches.sum() / actual_k)
                )
                recall_accum["RQ-Denoised-kNN"][ki].append(
                    float(matches.sum() / n_same) if n_same > 0 else 0.0
                )
            positions = np.where(neighbor_labels == q_label)[0]
            mrr_accum["RQ-Denoised-kNN"].append(
                float(1.0 / (positions[0] + 1)) if len(positions) > 0 else 0.0
            )

        # --- KM-Denoised-kNN ---
        km_queries = km_recon_test[batch_start:batch_end]
        kd_dists, kd_indices = km_recon_index.search(km_queries, max_k)
        for i in range(b_size):
            q_label = int(batch_labels[i])
            n_same = class_counts_train.get(q_label, 0)
            neighbor_labels = train_labels[kd_indices[i]]
            for ki, k in enumerate(k_values):
                actual_k = min(k, max_k)
                top_labels = neighbor_labels[:actual_k]
                matches = (top_labels == q_label).astype(np.float64)
                precision_accum["KM-Denoised-kNN"][ki].append(
                    float(matches.sum() / actual_k)
                )
                recall_accum["KM-Denoised-kNN"][ki].append(
                    float(matches.sum() / n_same) if n_same > 0 else 0.0
                )
            positions = np.where(neighbor_labels == q_label)[0]
            mrr_accum["KM-Denoised-kNN"].append(
                float(1.0 / (positions[0] + 1)) if len(positions) > 0 else 0.0
            )

    # --- Hash-based methods (per-query, bucket lookup) ---
    for i in range(n_test):
        q_label = int(test_labels[i])
        n_same = class_counts_train.get(q_label, 0)
        q_emb = test_emb[i]

        # RQ-Hash-kNN
        rq_key = tuple(trunk_codes_test[i].tolist())
        rq_bucket = rq_hash.get(rq_key)
        if rq_bucket is not None and len(rq_bucket) > 0:
            bucket_embs = train_emb[rq_bucket]
            dists = np.sum((bucket_embs - q_emb) ** 2, axis=1)
            bucket_labels = train_labels[rq_bucket]
            for ki, k in enumerate(k_values):
                p, r, rr = _knn_from_dists(dists, bucket_labels, q_label, k, n_same)
                precision_accum["RQ-Hash-kNN"][ki].append(p)
                recall_accum["RQ-Hash-kNN"][ki].append(r)
            # MRR uses max_k
            _, _, rr = _knn_from_dists(
                dists, bucket_labels, q_label, min(max_k, len(dists)), n_same
            )
            mrr_accum["RQ-Hash-kNN"].append(rr)
        else:
            for ki in range(len(k_values)):
                precision_accum["RQ-Hash-kNN"][ki].append(0.0)
                recall_accum["RQ-Hash-kNN"][ki].append(0.0)
            mrr_accum["RQ-Hash-kNN"].append(0.0)

        # KM-Hash-kNN
        km_key = (int(km_labels_test[i]),)
        km_bucket = km_hash.get(km_key)
        if km_bucket is not None and len(km_bucket) > 0:
            bucket_embs = train_emb[km_bucket]
            dists = np.sum((bucket_embs - q_emb) ** 2, axis=1)
            bucket_labels = train_labels[km_bucket]
            for ki, k in enumerate(k_values):
                p, r, rr = _knn_from_dists(dists, bucket_labels, q_label, k, n_same)
                precision_accum["KM-Hash-kNN"][ki].append(p)
                recall_accum["KM-Hash-kNN"][ki].append(r)
            _, _, rr = _knn_from_dists(
                dists, bucket_labels, q_label, min(max_k, len(dists)), n_same
            )
            mrr_accum["KM-Hash-kNN"].append(rr)
        else:
            for ki in range(len(k_values)):
                precision_accum["KM-Hash-kNN"][ki].append(0.0)
                recall_accum["KM-Hash-kNN"][ki].append(0.0)
            mrr_accum["KM-Hash-kNN"].append(0.0)

    # Aggregate
    results: dict[str, dict[str, list[float] | float]] = {}
    for m in METHOD_NAMES:
        results[m] = {
            "precision_at_k": [
                float(np.mean(precision_accum[m][ki])) for ki in range(len(k_values))
            ],
            "recall_at_k": [
                float(np.mean(recall_accum[m][ki])) for ki in range(len(k_values))
            ],
            "mrr": float(np.mean(mrr_accum[m])),
        }

    return results


def evaluate_few_shot(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    trunk_codes_train: np.ndarray,
    trunk_codes_test: np.ndarray,
    trunk_recon_train: np.ndarray,
    trunk_recon_test: np.ndarray,
    km_labels_train: np.ndarray,
    km_labels_test: np.ndarray,
    km_recon_train: np.ndarray,
    km_recon_test: np.ndarray,
    n_shot: int,
    k: int,
    rng: np.random.RandomState,
) -> dict[str, float]:
    """Evaluate few-shot classification accuracy for all methods.

    Samples n_shot examples per class as support set, then classifies
    each test query by majority vote of k-NN in the support set.

    Returns:
        method_name -> accuracy
    """
    import faiss as faiss_lib

    unique_classes = np.unique(train_labels)
    # Sample support set: n_shot per class
    support_indices: list[int] = []
    for cls in unique_classes:
        cls_indices = np.where(train_labels == cls)[0]
        n_available = min(n_shot, len(cls_indices))
        chosen = rng.choice(cls_indices, n_available, replace=False)
        support_indices.extend(chosen.tolist())
    support_indices_arr = np.array(support_indices, dtype=np.int64)

    support_emb = train_emb[support_indices_arr]
    support_labels = train_labels[support_indices_arr]
    support_trunk_codes = trunk_codes_train[support_indices_arr]
    support_trunk_recon = trunk_recon_train[support_indices_arr]
    support_km_labels = km_labels_train[support_indices_arr]
    support_km_recon = km_recon_train[support_indices_arr]

    n_test = test_emb.shape[0]
    actual_k = min(k, len(support_indices_arr))

    results: dict[str, float] = {}

    # --- BruteForce-kNN on support ---
    bf_index = faiss_lib.IndexFlatL2(support_emb.shape[1])
    bf_index.add(support_emb)
    _, bf_indices = bf_index.search(test_emb, actual_k)
    bf_preds = np.array(
        [
            Counter(support_labels[bf_indices[i]].tolist()).most_common(1)[0][0]
            for i in range(n_test)
        ]
    )
    results["BruteForce-kNN"] = float((bf_preds == test_labels).mean())

    # --- RQ-Denoised-kNN on support ---
    recon_index = faiss_lib.IndexFlatL2(support_trunk_recon.shape[1])
    recon_index.add(support_trunk_recon)
    _, rd_indices = recon_index.search(trunk_recon_test, actual_k)
    rd_preds = np.array(
        [
            Counter(support_labels[rd_indices[i]].tolist()).most_common(1)[0][0]
            for i in range(n_test)
        ]
    )
    results["RQ-Denoised-kNN"] = float((rd_preds == test_labels).mean())

    # --- KM-Denoised-kNN on support ---
    km_index = faiss_lib.IndexFlatL2(support_km_recon.shape[1])
    km_index.add(support_km_recon)
    _, kd_indices = km_index.search(km_recon_test, actual_k)
    kd_preds = np.array(
        [
            Counter(support_labels[kd_indices[i]].tolist()).most_common(1)[0][0]
            for i in range(n_test)
        ]
    )
    results["KM-Denoised-kNN"] = float((kd_preds == test_labels).mean())

    # --- RQ-Hash-kNN on support ---
    rq_hash = _build_hash_index(support_trunk_codes)
    rq_preds_list: list[int] = []
    for i in range(n_test):
        rq_key = tuple(trunk_codes_test[i].tolist())
        bucket = rq_hash.get(rq_key)
        if bucket is not None and len(bucket) > 0:
            bucket_embs = support_emb[bucket]
            dists = np.sum((bucket_embs - test_emb[i]) ** 2, axis=1)
            bk = min(actual_k, len(dists))
            if bk >= len(dists):
                topk_idx = np.argsort(dists)[:bk]
            else:
                topk_idx = np.argpartition(dists, bk)[:bk]
            topk_labels = support_labels[bucket[topk_idx]]
            pred = Counter(topk_labels.tolist()).most_common(1)[0][0]
        else:
            # Fallback: random class from support
            pred = int(rng.choice(unique_classes))
        rq_preds_list.append(pred)
    results["RQ-Hash-kNN"] = float((np.array(rq_preds_list) == test_labels).mean())

    # --- KM-Hash-kNN on support ---
    km_hash = _build_hash_index(support_km_labels.reshape(-1, 1))
    km_preds_list: list[int] = []
    for i in range(n_test):
        km_key = (int(km_labels_test[i]),)
        bucket = km_hash.get(km_key)
        if bucket is not None and len(bucket) > 0:
            bucket_embs = support_emb[bucket]
            dists = np.sum((bucket_embs - test_emb[i]) ** 2, axis=1)
            bk = min(actual_k, len(dists))
            if bk >= len(dists):
                topk_idx = np.argsort(dists)[:bk]
            else:
                topk_idx = np.argpartition(dists, bk)[:bk]
            topk_labels = support_labels[bucket[topk_idx]]
            pred = Counter(topk_labels.tolist()).most_common(1)[0][0]
        else:
            pred = int(rng.choice(unique_classes))
        km_preds_list.append(pred)
    results["KM-Hash-kNN"] = float((np.array(km_preds_list) == test_labels).mean())

    # --- Faiss-IVF on support (just use brute-force on small support) ---
    # For few-shot, support is tiny so IVF doesn't make sense; use brute-force
    results["Faiss-IVF"] = results["BruteForce-kNN"]

    return results


# =============================================================================
# Main experiment
# =============================================================================


def run_retrieval_experiment(config: RetrievalConfig) -> dict[str, object]:
    """End-to-end retrieval / few-shot experiment."""
    import faiss

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading dataset: {config.dataset}")
    train_dataset, _val_dataset, test_dataset, metadata = load_dataset(config.dataset)

    n_classes = metadata.n_classes
    logger.info(
        f"Loaded: {metadata.n_samples} samples, "
        f"{metadata.n_continuous} cont, {metadata.n_categorical} cat, "
        f"{n_classes} classes"
    )

    per_seed_results: list[dict[str, object]] = []

    for enc_idx in range(config.n_encoder_seeds):
        encoder_seed = config.seed + enc_idx * 10000
        logger.info(
            f"\n  Encoder seed {enc_idx + 1}/{config.n_encoder_seeds} "
            f"(seed={encoder_seed})"
        )

        np.random.seed(encoder_seed)
        torch.manual_seed(encoder_seed)

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

        # ----- Expensive precompute -----
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

        logger.info("  Collecting train embeddings...")
        train_emb, train_labels = collect_embeddings_and_labels(
            encoder, train_loader, device
        )
        logger.info(f"  Train embeddings: {train_emb.shape}")

        logger.info("  Collecting test embeddings...")
        test_emb, test_labels = collect_embeddings_and_labels(
            encoder, test_loader, device
        )
        logger.info(f"  Test embeddings: {test_emb.shape}")

        # Fit RQ
        logger.info("  Fitting RQ trunk codes...")
        trunk_codes_train, rq = compute_trunk_codes(
            train_emb, config.d_cut, config.nbits
        )
        n_unique_train = len(torch.unique(trunk_codes_train, dim=0))
        logger.info(
            f"  Trunk codes train: {trunk_codes_train.shape}, unique: {n_unique_train}"
        )

        train_emb_np = train_emb.detach().numpy().astype(np.float32)
        test_emb_np = test_emb.detach().numpy().astype(np.float32)
        train_labels_np = train_labels.numpy()
        test_labels_np = test_labels.numpy()

        # Trunk codes for test
        trunk_codes_test_raw = rq.compute_codes(test_emb_np)  # pyre-ignore[16]
        trunk_codes_test = trunk_codes_test_raw.astype(np.int64)
        trunk_codes_train_np = trunk_codes_train.numpy()
        n_unique_test = len(np.unique(trunk_codes_test, axis=0))
        logger.info(f"  Unique trunk codes (test): {n_unique_test}")

        # Trunk reconstructions
        trunk_codes_train_raw = rq.compute_codes(train_emb_np)  # pyre-ignore[16]
        trunk_recon_train: np.ndarray = rq.decode(
            trunk_codes_train_raw
        )  # pyre-ignore[16]
        trunk_recon_test: np.ndarray = rq.decode(
            trunk_codes_test_raw
        )  # pyre-ignore[16]

        # KMeans
        km_n_clusters = config.km_clusters or (2**config.nbits)
        km = MiniBatchKMeans(
            n_clusters=min(km_n_clusters, train_emb_np.shape[0]),
            random_state=encoder_seed,
            n_init=3,
        )
        km.fit(train_emb_np)
        km_labels_train = km.labels_
        km_labels_test = km.predict(test_emb_np)
        km_recon_train = km.cluster_centers_[km_labels_train]
        km_recon_test = km.cluster_centers_[km_labels_test]

        # Faiss IVF index on train embeddings
        dim = train_emb_np.shape[1]
        n_train = train_emb_np.shape[0]
        nlist = min(int(np.sqrt(n_train)), 256)
        quantizer = faiss.IndexFlatL2(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        ivf_index.train(train_emb_np)
        ivf_index.add(train_emb_np)
        ivf_index.nprobe = min(nlist, 32)

        # D_cut diagnostic: H(Y|Z)
        h_y_given_z_train = conditional_entropy(
            train_labels_np, trunk_codes_train_np, alpha=1.0
        )
        logger.info(f"  H(Y|trunk_codes) on train: {h_y_given_z_train:.4f}")

        # ----- Retrieval evaluation -----
        logger.info("  Evaluating retrieval methods...")
        retrieval_results = evaluate_retrieval(
            train_emb=train_emb_np,
            test_emb=test_emb_np,
            train_labels=train_labels_np,
            test_labels=test_labels_np,
            trunk_codes_train=trunk_codes_train_np,
            trunk_codes_test=trunk_codes_test,
            trunk_recon_train=trunk_recon_train.astype(np.float32),
            trunk_recon_test=trunk_recon_test.astype(np.float32),
            km_labels_train=km_labels_train,
            km_labels_test=km_labels_test,
            km_recon_train=km_recon_train.astype(np.float32),
            km_recon_test=km_recon_test.astype(np.float32),
            faiss_ivf=ivf_index,
            k_values=config.k_values,
        )

        # Log retrieval results
        logger.info("\n  Retrieval results:")
        header = (
            f"  {'Method':<20s}"
            + "".join(f"  P@{k:<3d}" for k in config.k_values)
            + "    MRR"
        )
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for m in METHOD_NAMES:
            prec = retrieval_results[m]["precision_at_k"]
            mrr = retrieval_results[m]["mrr"]
            row = f"  {m:<20s}"
            for p in prec:
                row += f"  {p:.3f}"
            row += f"  {mrr:.3f}"
            logger.info(row)

        # ----- Few-shot evaluation -----
        logger.info("\n  Evaluating few-shot classification...")
        few_shot_results: dict[str, dict[str, float]] = {}
        few_shot_k = min(5, max(config.k_values))  # Use k=5 for few-shot voting

        for n_shot in config.n_shot_values:
            fs_rng = np.random.RandomState(encoder_seed + n_shot)
            fs_result = evaluate_few_shot(
                train_emb=train_emb_np,
                test_emb=test_emb_np,
                train_labels=train_labels_np,
                test_labels=test_labels_np,
                trunk_codes_train=trunk_codes_train_np,
                trunk_codes_test=trunk_codes_test,
                trunk_recon_train=trunk_recon_train.astype(np.float32),
                trunk_recon_test=trunk_recon_test.astype(np.float32),
                km_labels_train=km_labels_train,
                km_labels_test=km_labels_test,
                km_recon_train=km_recon_train.astype(np.float32),
                km_recon_test=km_recon_test.astype(np.float32),
                n_shot=n_shot,
                k=few_shot_k,
                rng=fs_rng,
            )
            few_shot_results[f"{n_shot}-shot"] = fs_result

            parts = ", ".join(f"{m}={v:.3f}" for m, v in fs_result.items())
            logger.info(f"    {n_shot}-shot: {parts}")

        seed_result: dict[str, object] = {
            "encoder_seed": encoder_seed,
            "n_train": int(train_emb_np.shape[0]),
            "n_test": int(test_emb_np.shape[0]),
            "n_unique_trunk_codes_train": int(n_unique_train),
            "n_unique_trunk_codes_test": int(n_unique_test),
            "h_y_given_z_train": h_y_given_z_train,
            "methods": retrieval_results,
            "few_shot": few_shot_results,
        }
        per_seed_results.append(seed_result)

    # ----- Aggregate across seeds -----
    logger.info("\n" + "=" * 80)
    logger.info(
        f"AGGREGATE RESULTS: {config.dataset} (d_cut={config.d_cut}, "
        f"nbits={config.nbits}, {config.n_encoder_seeds} seeds)"
    )
    logger.info("=" * 80)

    aggregate: dict[str, object] = {}

    # Aggregate retrieval metrics
    for m in METHOD_NAMES:
        prec_arrays = np.array(
            [s["methods"][m]["precision_at_k"] for s in per_seed_results]
        )
        recall_arrays = np.array(
            [s["methods"][m]["recall_at_k"] for s in per_seed_results]
        )
        mrr_values = [s["methods"][m]["mrr"] for s in per_seed_results]

        m_key = m.lower().replace("-", "_")
        aggregate[f"{m_key}_precision_at_k_mean"] = prec_arrays.mean(axis=0).tolist()
        aggregate[f"{m_key}_precision_at_k_std"] = prec_arrays.std(axis=0).tolist()
        aggregate[f"{m_key}_recall_at_k_mean"] = recall_arrays.mean(axis=0).tolist()
        aggregate[f"{m_key}_recall_at_k_std"] = recall_arrays.std(axis=0).tolist()
        aggregate[f"{m_key}_mrr_mean"] = float(np.mean(mrr_values))
        aggregate[f"{m_key}_mrr_std"] = float(np.std(mrr_values))

    # Aggregate few-shot
    for n_shot in config.n_shot_values:
        shot_key = f"{n_shot}-shot"
        for m in METHOD_NAMES:
            vals = [s["few_shot"][shot_key][m] for s in per_seed_results]
            m_key = m.lower().replace("-", "_")
            aggregate[f"few_shot_{n_shot}_{m_key}_mean"] = float(np.mean(vals))
            aggregate[f"few_shot_{n_shot}_{m_key}_std"] = float(np.std(vals))

    # Log aggregate precision table
    logger.info("\nPrecision@k (mean +/- std):")
    header = (
        f"{'Method':<20s}"
        + "".join(f"  P@{k:<3d}" for k in config.k_values)
        + "    MRR"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for m in METHOD_NAMES:
        m_key = m.lower().replace("-", "_")
        prec_mean = aggregate[f"{m_key}_precision_at_k_mean"]
        prec_std = aggregate[f"{m_key}_precision_at_k_std"]
        mrr_mean = aggregate[f"{m_key}_mrr_mean"]
        mrr_std = aggregate[f"{m_key}_mrr_std"]
        row = f"{m:<20s}"
        for pm, ps in zip(prec_mean, prec_std):
            row += f"  {pm:.3f}"
        row += f"  {mrr_mean:.3f}"
        logger.info(row)

    # Log few-shot table
    logger.info("\nFew-shot accuracy (mean +/- std):")
    fs_header = f"{'Method':<20s}" + "".join(
        f"  {n}shot " for n in config.n_shot_values
    )
    logger.info(fs_header)
    logger.info("-" * len(fs_header))
    for m in METHOD_NAMES:
        m_key = m.lower().replace("-", "_")
        row = f"{m:<20s}"
        for n_shot in config.n_shot_values:
            mean = aggregate[f"few_shot_{n_shot}_{m_key}_mean"]
            std = aggregate[f"few_shot_{n_shot}_{m_key}_std"]
            row += f"  {mean:.3f}"
        logger.info(row)

    result: dict[str, object] = {
        "dataset": config.dataset,
        "n_classes": n_classes,
        "config": asdict(config),
        "per_seed_results": per_seed_results,
        "aggregate": aggregate,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval / few-shot classification experiment: "
        "RQ trunk codes as label-aware hash buckets"
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
        help="Number of encoder/RQ random seeds",
    )
    parser.add_argument(
        "--d-cut",
        type=int,
        default=2,
        help="Trunk depth for RQ codes",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=6,
        help="Bits per RQ level",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Encoder embedding dimension",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="k values for precision/recall@k",
    )
    parser.add_argument(
        "--n-shot-values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="n-shot support set sizes for few-shot evaluation",
    )
    parser.add_argument(
        "--km-clusters",
        type=int,
        default=None,
        help="Number of KMeans clusters (default: 2^nbits)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/retrieval_results",
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
    args = parser.parse_args()

    config = RetrievalConfig(
        dataset=args.dataset,
        n_encoder_seeds=args.n_encoder_seeds,
        d_cut=args.d_cut,
        nbits=args.nbits,
        embedding_dim=args.embedding_dim,
        k_values=args.k_values,
        n_shot_values=args.n_shot_values,
        km_clusters=args.km_clusters,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
    )

    logger.info("=" * 60)
    logger.info("RETRIEVAL / FEW-SHOT CLASSIFICATION EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset}")
    logger.info(
        f"n_encoder_seeds: {config.n_encoder_seeds}, d_cut: {config.d_cut}, "
        f"nbits: {config.nbits}"
    )
    logger.info(f"k_values: {config.k_values}, n_shot_values: {config.n_shot_values}")

    start_time = time.time()
    results = run_retrieval_experiment(config)
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_dcut{config.d_cut}" if config.d_cut != 2 else ""
    output_file = output_dir / f"retrieval_{config.dataset}{suffix}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
