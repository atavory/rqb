#!/usr/bin/env python3


"""Multi-dataset m-RQ experiments for paper inclusion.

This script runs m-RQ vs baselines across all datasets and logs results
in a structured format for reproducible paper tables.

Example:
    python3 scripts/experiments/run_all_experiments.py -- \
        --datasets adult bank_marketing german_credit \
        --seeds 42 123 456

Output format (JSON):
    {
        "metadata": {
            "timestamp": "2024-02-07T08:45:00Z",
            "git_hash": "abc123",
            "config": {...}
        },
        "results": [
            {"dataset": "adult", "seed": 42, "method": "m-RQ", "accuracy": 0.847, ...},
            ...
        ],
        "summary": {
            "adult": {"m-RQ": {"mean": 0.847, "std": 0.003}, ...},
            ...
        }
    }
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import logging
from modules.contrastive.faiss_mrq_module import (
    FaissMRQConfig,
    FaissMRQModule,
    find_optimal_d_cut,
)
from modules.contrastive.losses import (
    hard_neg_supcon_loss,
    info_nce_loss,
    supcon_loss,
    supcon_proto_loss,
    weighted_supcon_loss,
)
from modules.contrastive.projection import ProjectionHead
from modules.data import (
    load_dataset,
    TabularDataset,
    TabularDatasetMetadata,
)
from modules.embeddings import (
    EmbeddingConfig,
    EmbeddingExtractor,
)
from modules.encoders.tab_transformer import MLPEncoder
from modules.features import (
    collect_raw_features,
    FeatureCache,
    normalize_for_rq,
    select_features_unsupervised,
)
from modules.significance import (
    compute_pairwise_significance,
    log_significance,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# All RQ-based methods (ours) — all names contain "RQ" for paper clarity
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
class ExperimentConfig:
    """Configuration for multi-dataset experiments."""

    # Datasets
    datasets: list[str] = field(
        default_factory=lambda: ["adult", "bank_marketing", "german_credit"]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    # Faiss RQ params
    n_levels: int = 8
    nbits: int = 6
    d_cut: int = 4
    beam_size: int = 5
    use_gpu: bool = True
    auto_d_cut: bool = False

    # Encoder params
    embedding_dim: int = 64
    hidden_dim: int = 128

    # Training params
    encoder_epochs: int = 20
    contrastive_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    temperature: float = 0.07
    projection_dim: int = 32
    view_refresh_interval: int = 25  # Refresh m-RQ views every N epochs
    mrq_noise_scale: float = 0.1  # Noise added to m-RQ views for diversity

    # Hybrid RQ+supervised method params
    prototype_weight: float = 0.5  # Weight for prototype attraction in RQ-SupCon+Proto
    hard_neg_strength: float = 0.5  # Hard negative up-weighting in RQ-HardNegSupCon
    include_hybrid_methods: bool = (
        False  # Include RQ-SupCon+Proto, RQ-WeightedSupCon, etc.
    )
    methods_filter: list[str] | None = None  # Run only these methods (None = all)

    # Feature mode: "embeddings" (default), "raw", or "selected"
    feature_mode: str = "embeddings"
    # Cache directory for raw features and feature selection results
    feature_cache_dir: str = "results/feature_cache"
    # Embedding type: "raw", "lgbm", or "iforest"
    embedding_type: str = "raw"


@dataclass
class SingleRunResult:
    """Result from a single experiment run."""

    dataset: str
    seed: int
    method: str
    accuracy: float
    time_seconds: float
    codebook_time_seconds: float = 0.0
    d_cut: int | None = None
    reconstruction_error: float | None = None


def get_git_hash() -> str:
    """Get current git/hg commit hash."""
    try:
        result = subprocess.run(
            ["hg", "log", "-r", ".", "--template", "{node|short}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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
    train_loader: DataLoader,
    n_classes: int,
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    """Pre-train encoder with classification loss."""
    classifier = nn.Linear(config.embedding_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    encoder.train()
    for _epoch in range(config.encoder_epochs):
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

    return encoder


def mrq_contrastive_loss(
    embeddings: Tensor,
    views: Tensor,
    labels: Tensor,
    projection_head: nn.Module,
    temperature: float,
    noise_scale: float = 0.05,
) -> Tensor:
    """Compute label-aware m-RQ contrastive loss.

    Only uses the correct class's tail view as the positive, rather than
    treating all m class views as positives (which blurs class boundaries).
    Adds stochastic noise for augmentation diversity since RQ reconstruction
    is deterministic.
    """
    batch_idx = torch.arange(embeddings.shape[0], device=embeddings.device)
    correct_views = views[batch_idx, labels, :]  # (B, dim) — correct class only

    # Add stochastic noise for augmentation diversity
    if noise_scale > 0:
        correct_views = correct_views + torch.randn_like(correct_views) * noise_scale

    z = projection_head(embeddings)
    z_view = projection_head(correct_views)
    return info_nce_loss(z, z_view, temperature)


def mrq_gaussian_hybrid_loss(
    embeddings: Tensor,
    views: Tensor,
    labels: Tensor,
    projection_head: nn.Module,
    temperature: float,
    noise_scale: float = 0.1,
) -> Tensor:
    """Combine m-RQ structural views with Gaussian stochastic augmentation.

    Uses two positive pairs:
    1. (original, m-RQ correct class view) — structural augmentation
    2. (original, Gaussian noise) — stochastic augmentation
    Loss = 0.5 * InfoNCE(orig, mrq_view) + 0.5 * InfoNCE(orig, gaussian)

    This hybrid addresses m-RQ's determinism problem: RQ views provide
    class-aware structure while Gaussian noise provides stochastic diversity.
    """
    batch_idx = torch.arange(embeddings.shape[0], device=embeddings.device)
    correct_views = views[batch_idx, labels, :]

    # Structural positive: m-RQ view (no added noise — let RQ structure speak)
    z = projection_head(embeddings)
    z_mrq = projection_head(correct_views)
    l_mrq = info_nce_loss(z, z_mrq, temperature)

    # Stochastic positive: Gaussian augmentation
    aug = embeddings + torch.randn_like(embeddings) * noise_scale
    z_gauss = projection_head(aug)
    l_gauss = info_nce_loss(z, z_gauss, temperature)

    return 0.5 * l_mrq + 0.5 * l_gauss


def collect_embeddings(
    encoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect all embeddings and labels from the train loader."""
    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_embeddings.append(emb)
            all_labels.append(batch_y.to(device))
    return torch.cat(all_embeddings), torch.cat(all_labels)


def _compute_prototypes(embeddings: Tensor, labels: Tensor) -> Tensor:
    """Compute per-class prototype centroids from embeddings.

    Args:
        embeddings: (N, dim) tensor.
        labels: (N,) tensor of class labels.

    Returns:
        prototypes: (n_classes, dim) tensor of class centroids.
    """
    n_classes = int(labels.max().item()) + 1
    dim = embeddings.shape[1]
    prototypes = torch.zeros(n_classes, dim, device=embeddings.device)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(dim=0)
    return prototypes


def _compute_recon_quality(embeddings: Tensor, views: Tensor, labels: Tensor) -> Tensor:
    """Compute per-sample reconstruction quality from m-RQ views.

    Quality = exp(-||embedding - correct_view||^2), in (0, 1].

    Args:
        embeddings: (N, dim) tensor.
        views: (N, n_classes, dim) tensor of m-RQ views.
        labels: (N,) tensor of class labels.

    Returns:
        quality: (N,) tensor in (0, 1].
    """
    batch_idx = torch.arange(len(labels), device=embeddings.device)
    correct_views = views[batch_idx, labels, :]
    recon_err = (embeddings - correct_views).pow(2).sum(dim=-1)
    return torch.exp(-recon_err)


def train_contrastive(
    encoder: nn.Module,
    train_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    method: str,
    views: Tensor | None = None,
    labels: Tensor | None = None,
    mrq: FaissMRQModule | None = None,
    all_cont_features: Tensor | None = None,
    all_cat_features: Tensor | None = None,
    prototypes: Tensor | None = None,
    recon_quality: Tensor | None = None,
) -> tuple[nn.Module, float]:
    """Train encoder with contrastive loss (various methods).

    RQ-based methods (ours): m-RQ, m-RQ+Gaussian, RQ-SupCon+Proto,
        RQ-WeightedSupCon, RQ-HardNegSupCon, RQ-Proto+Weighted.
    Baselines: Gaussian, Dropout, SCARF, SupCon.
    """
    projection_head = ProjectionHead(
        input_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.projection_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    encoder.train()
    projection_head.train()

    start_time = time.time()

    if (
        method in RQ_METHODS
        and views is not None
        and labels is not None
        and all_cont_features is not None
    ):
        # RQ-based methods: use raw features + pre-computed views
        if all_cat_features is not None:
            mrq_dataset = TensorDataset(
                all_cont_features, all_cat_features, views, labels
            )
        else:
            dummy_cat = torch.zeros(len(all_cont_features), 0, dtype=torch.long)
            mrq_dataset = TensorDataset(all_cont_features, dummy_cat, views, labels)
        mrq_loader = DataLoader(mrq_dataset, batch_size=config.batch_size, shuffle=True)
        has_cat = all_cat_features is not None

        # Prototype tensor on device (for SupCon+Proto / Proto+Weighted)
        proto_t = prototypes.to(device) if prototypes is not None else None
        # Recon quality tensor (for WeightedSupCon / Proto+Weighted)
        rq_t = recon_quality.to(device) if recon_quality is not None else None

        view_refresh_interval = config.view_refresh_interval

        for epoch in range(config.contrastive_epochs):
            # Periodically refresh views as the encoder changes
            if (
                view_refresh_interval > 0
                and epoch > 0
                and epoch % view_refresh_interval == 0
                and mrq is not None
            ):
                encoder.eval()
                with torch.no_grad():
                    new_emb_list: list[Tensor] = []
                    for i in range(0, len(all_cont_features), config.batch_size):
                        b_cont = all_cont_features[i : i + config.batch_size].to(device)
                        b_cat = (
                            all_cat_features[i : i + config.batch_size].to(device)
                            if has_cat
                            else None
                        )
                        emb = encoder(cat_features=b_cat, cont_features=b_cont)
                        new_emb_list.append(emb.cpu())
                    new_embeddings = torch.cat(new_emb_list)
                    # Ensure labels on same device as new_embeddings (CPU)
                    # to avoid device mismatch in indexing operations.
                    labels_cpu = labels.cpu()
                    mrq.fit(new_embeddings, labels_cpu)
                    new_views = mrq.generate_views(new_embeddings)
                    # Update prototypes from new embeddings
                    if proto_t is not None:
                        proto_t = _compute_prototypes(new_embeddings, labels_cpu).to(
                            device
                        )
                    # Update recon quality
                    if rq_t is not None:
                        rq_t = _compute_recon_quality(
                            new_embeddings, new_views, labels_cpu
                        ).to(device)
                if has_cat:
                    mrq_dataset = TensorDataset(
                        all_cont_features, all_cat_features, new_views, labels
                    )
                else:
                    mrq_dataset = TensorDataset(
                        all_cont_features, dummy_cat, new_views, labels
                    )
                mrq_loader = DataLoader(
                    mrq_dataset, batch_size=config.batch_size, shuffle=True
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                encoder.train()

            for batch_idx_data in mrq_loader:
                batch_cont = batch_idx_data[0].to(device)
                batch_cat_raw = batch_idx_data[1]
                batch_views = batch_idx_data[2].to(device)
                batch_labels = batch_idx_data[3].to(device)
                cat_feat = batch_cat_raw.to(device) if has_cat else None

                live_embeddings = encoder(
                    cat_features=cat_feat, cont_features=batch_cont
                )

                if method == "m-RQ":
                    loss = mrq_contrastive_loss(
                        live_embeddings,
                        batch_views,
                        batch_labels,
                        projection_head,
                        config.temperature,
                        noise_scale=config.mrq_noise_scale,
                    )
                elif method == "m-RQ+Gaussian":
                    loss = mrq_gaussian_hybrid_loss(
                        live_embeddings,
                        batch_views,
                        batch_labels,
                        projection_head,
                        config.temperature,
                        noise_scale=config.mrq_noise_scale,
                    )
                elif method == "RQ-SupCon+Proto":
                    z = projection_head(live_embeddings)
                    loss = supcon_proto_loss(
                        z,
                        batch_labels,
                        projection_head(proto_t) if proto_t is not None else z,
                        temperature=config.temperature,
                        prototype_weight=config.prototype_weight,
                    )
                elif method == "RQ-WeightedSupCon":
                    z = projection_head(live_embeddings)
                    # Compute per-sample recon quality from batch views
                    batch_idx = torch.arange(len(batch_labels), device=device)
                    correct_views = batch_views[batch_idx, batch_labels, :]
                    batch_recon_err = (
                        (live_embeddings.detach() - correct_views).pow(2).mean(dim=-1)
                    )
                    # Normalize to [0, 1] within batch for stability
                    max_err = batch_recon_err.max() + 1e-8
                    batch_rq = 1.0 - batch_recon_err / max_err  # (0, 1]
                    loss = weighted_supcon_loss(
                        z,
                        batch_labels,
                        batch_rq,
                        temperature=config.temperature,
                    )
                elif method == "RQ-HardNegSupCon":
                    z = projection_head(live_embeddings)
                    loss = hard_neg_supcon_loss(
                        z,
                        batch_labels,
                        temperature=config.temperature,
                        hard_neg_strength=config.hard_neg_strength,
                    )
                elif method == "RQ-Proto+Weighted":
                    z = projection_head(live_embeddings)
                    # Combined: prototype + weighted SupCon
                    batch_idx = torch.arange(len(batch_labels), device=device)
                    correct_views = batch_views[batch_idx, batch_labels, :]
                    batch_recon_err = (
                        (live_embeddings.detach() - correct_views).pow(2).mean(dim=-1)
                    )
                    max_err = batch_recon_err.max() + 1e-8
                    batch_rq = 1.0 - batch_recon_err / max_err
                    l_weighted = weighted_supcon_loss(
                        z,
                        batch_labels,
                        batch_rq,
                        temperature=config.temperature,
                    )
                    l_proto = supcon_proto_loss(
                        z,
                        batch_labels,
                        projection_head(proto_t) if proto_t is not None else z,
                        temperature=config.temperature,
                        prototype_weight=config.prototype_weight,
                    )
                    loss = 0.5 * l_weighted + 0.5 * l_proto
                else:
                    loss = mrq_contrastive_loss(
                        live_embeddings,
                        batch_views,
                        batch_labels,
                        projection_head,
                        config.temperature,
                        noise_scale=config.mrq_noise_scale,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    else:
        # Standard augmentation-based baseline methods
        noise_scale, dropout_rate = {
            "Gaussian": (0.1, 0.0),
            "Dropout": (0.0, 0.3),
            "SCARF": (0.1, 0.3),
        }.get(method, (0.1, 0.0))

        for _epoch in range(config.contrastive_epochs):
            for batch_features, batch_y in train_loader:
                cont_features, cat_features = get_features(batch_features, device)
                batch_y = batch_y.to(device)

                embeddings = encoder(
                    cat_features=cat_features, cont_features=cont_features
                )

                if method == "Gaussian":
                    noise = torch.randn_like(embeddings) * noise_scale
                    aug_embeddings = embeddings + noise
                elif method == "Dropout":
                    dropout = nn.Dropout(dropout_rate)
                    aug_embeddings = dropout(embeddings)
                elif method == "SCARF":
                    mask = torch.rand_like(embeddings) < dropout_rate
                    noise = torch.randn_like(embeddings) * noise_scale
                    aug_embeddings = torch.where(mask, noise, embeddings)
                elif method == "SupCon":
                    z = projection_head(embeddings)
                    loss = supcon_loss(z, batch_y, temperature=config.temperature)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    continue
                else:
                    aug_embeddings = embeddings + torch.randn_like(embeddings) * 0.1

                z = projection_head(embeddings)
                z_aug = projection_head(aug_embeddings)
                loss = info_nce_loss(z, z_aug, config.temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return encoder, time.time() - start_time


def linear_probe_eval(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate encoder using linear probe."""
    encoder.eval()

    train_embeddings: list[Tensor] = []
    train_labels: list[Tensor] = []
    with torch.no_grad():
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            train_embeddings.append(emb.cpu())
            train_labels.append(batch_y)

    X_train = torch.cat(train_embeddings).numpy()
    y_train = torch.cat(train_labels).numpy()

    test_embeddings: list[Tensor] = []
    test_labels: list[Tensor] = []
    with torch.no_grad():
        for batch_features, batch_y in test_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            test_embeddings.append(emb.cpu())
            test_labels.append(batch_y)

    X_test = torch.cat(test_embeddings).numpy()
    y_test = torch.cat(test_labels).numpy()

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return float(accuracy_score(y_test, y_pred))


def run_single_experiment(
    dataset_name: str,
    seed: int,
    method: str,
    config: ExperimentConfig,
    device: torch.device,
    train_dataset: TabularDataset,
    test_dataset: TabularDataset,
    metadata: TabularDatasetMetadata,
) -> SingleRunResult:
    """Run a single experiment with specific dataset, seed, method."""
    logger.info(
        f"[PHASE] run_single_experiment: dataset={dataset_name}, seed={seed}, "
        f"method={method}"
    )

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create encoder
    encoder = MLPEncoder(
        n_categories=metadata.category_sizes,
        n_continuous=metadata.n_continuous,
        dim=config.embedding_dim,
        hidden_dims=[config.hidden_dim, config.hidden_dim],
    ).to(device)

    # Pre-train encoder
    logger.info(
        f"[PHASE] encoder_pretrain_start: epochs={config.encoder_epochs}, "
        f"n_classes={metadata.n_classes}"
    )
    encoder = pretrain_encoder(
        encoder, train_loader, metadata.n_classes, config, device
    )
    logger.info("[PHASE] encoder_pretrain_complete")

    start_time = time.time()
    codebook_time = 0.0
    d_cut_used = None
    recon_error = None
    views = None
    mrq = None
    all_labels_t = None
    all_cont_t = None
    all_cat_t = None
    prototypes = None
    recon_quality = None

    if method in RQ_METHODS:
        # All RQ-based methods need embeddings, views, and raw features
        logger.info("[PHASE] collecting_embeddings_start")
        collect_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False
        )
        encoder.eval()
        all_embeddings: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_cont_features: list[Tensor] = []
        all_cat_features: list[Tensor] = []
        has_cat_features = False
        with torch.no_grad():
            for batch_features, batch_y in collect_loader:
                cont_features, cat_features = get_features(batch_features, device)
                emb = encoder(cat_features=cat_features, cont_features=cont_features)
                all_embeddings.append(emb)
                all_labels.append(batch_y.to(device))
                all_cont_features.append(cont_features.cpu())
                if cat_features is not None:
                    all_cat_features.append(cat_features.cpu())
                    has_cat_features = True
        all_embeddings_t = torch.cat(all_embeddings)
        all_labels_t = torch.cat(all_labels)
        all_cont_t = torch.cat(all_cont_features)
        all_cat_t = torch.cat(all_cat_features) if has_cat_features else None
        logger.info(
            f"[PHASE] collecting_embeddings_complete: shape={list(all_embeddings_t.shape)}, "
            f"n_classes={int(all_labels_t.max().item()) + 1}"
        )

        # Auto d_cut if requested
        if config.auto_d_cut:
            d_cut_used, _ = find_optimal_d_cut(
                all_embeddings_t,
                all_labels_t,
                max_levels=config.n_levels,
                nbits=config.nbits,
                use_gpu=False,
            )
        else:
            d_cut_used = config.d_cut

        # Fit faiss m-RQ
        logger.info(
            f"[PHASE] rq_codebook_fit_start: n_levels={config.n_levels}, "
            f"nbits={config.nbits}, d_cut={d_cut_used}, "
            f"embedding_shape={list(all_embeddings_t.shape)}, "
            f"n_classes={metadata.n_classes}"
        )
        faiss_config = FaissMRQConfig(
            n_levels=config.n_levels,
            nbits=config.nbits,
            d_cut=d_cut_used,
            use_gpu=config.use_gpu,
            beam_size=config.beam_size,
        )
        mrq = FaissMRQModule(
            n_classes=metadata.n_classes,
            embedding_dim=config.embedding_dim,
            config=faiss_config,
        )

        codebook_start = time.time()
        mrq.fit(all_embeddings_t, all_labels_t)
        codebook_time = time.time() - codebook_start
        logger.info(f"[PHASE] rq_codebook_fit_complete: time={codebook_time:.1f}s")

        # Generate views
        logger.info("[PHASE] view_generation_start")
        with torch.no_grad():
            views = mrq.generate_views(all_embeddings_t)
        logger.info(f"[PHASE] view_generation_complete: shape={list(views.shape)}")

        recon_error = mrq.get_reconstruction_error(all_embeddings_t, all_labels_t)

        # Compute prototypes and recon quality for hybrid methods
        if method in {"RQ-SupCon+Proto", "RQ-Proto+Weighted"}:
            logger.info("[PHASE] prototype_compute_start")
            prototypes = _compute_prototypes(all_embeddings_t, all_labels_t)
            logger.info(
                f"[PHASE] prototype_compute_complete: shape={list(prototypes.shape)}"
            )
        if method in {"RQ-WeightedSupCon", "RQ-Proto+Weighted"}:
            logger.info("[PHASE] recon_quality_compute_start")
            recon_quality = _compute_recon_quality(
                all_embeddings_t, views, all_labels_t
            )
            logger.info(
                f"[PHASE] recon_quality_compute_complete: "
                f"shape={list(recon_quality.shape)}, "
                f"mean={recon_quality.mean().item():.4f}"
            )

    # Contrastive training
    logger.info(
        f"[PHASE] contrastive_train_start: method={method}, "
        f"epochs={config.contrastive_epochs}"
    )
    is_rq = method in RQ_METHODS
    encoder, _ = train_contrastive(
        encoder,
        train_loader,
        config,
        device,
        method,
        views,
        labels=all_labels_t if is_rq else None,
        mrq=mrq if is_rq else None,
        all_cont_features=all_cont_t if is_rq else None,
        all_cat_features=all_cat_t if is_rq else None,
        prototypes=prototypes,
        recon_quality=recon_quality,
    )

    total_time = time.time() - start_time
    logger.info(f"[PHASE] contrastive_train_complete: time={total_time:.1f}s")

    # Evaluate
    logger.info("[PHASE] linear_probe_eval_start")
    accuracy = linear_probe_eval(encoder, train_loader, test_loader, device)
    logger.info(f"[PHASE] linear_probe_eval_complete: accuracy={accuracy:.4f}")

    return SingleRunResult(
        dataset=dataset_name,
        seed=seed,
        method=method,
        accuracy=accuracy,
        time_seconds=total_time,
        codebook_time_seconds=codebook_time,
        d_cut=d_cut_used,
        reconstruction_error=recon_error,
    )


def run_all_experiments(config: ExperimentConfig) -> dict[str, Any]:
    """Run all experiments across datasets, seeds, and methods."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Methods to compare
    methods = ["m-RQ", "m-RQ+Gaussian", "Gaussian", "Dropout", "SCARF", "SupCon"]
    if config.include_hybrid_methods:
        methods.extend(
            [
                "RQ-SupCon+Proto",
                "RQ-WeightedSupCon",
                "RQ-HardNegSupCon",
                "RQ-Proto+Weighted",
            ]
        )
    if config.methods_filter is not None:
        methods = [m for m in methods if m in config.methods_filter]

    # Collect all results
    all_results: list[SingleRunResult] = []

    for dataset_name in config.datasets:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'=' * 60}")

        try:
            train_dataset, _val_dataset, test_dataset, metadata = load_dataset(
                dataset_name
            )
            logger.info(
                f"Loaded: {metadata.n_samples} samples, "
                f"{metadata.n_continuous} cont, {metadata.n_categorical} cat, "
                f"{metadata.n_classes} classes"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        for seed in config.seeds:
            # Pre-train encoder ONCE per (dataset, seed) — deterministic from seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # If using tree embeddings, replace the input features
            if config.embedding_type != "raw":
                logger.info(
                    f"  Extracting {config.embedding_type} embeddings for encoder input..."
                )
                raw_feats, raw_labels = collect_raw_features(train_dataset)
                test_feats, test_labels = collect_raw_features(test_dataset)
                extractor = EmbeddingExtractor(
                    EmbeddingConfig(embedding_type=config.embedding_type)
                )
                y_fit = raw_labels if config.embedding_type == "lgbm" else None
                emb_train = extractor.fit_transform(raw_feats, y_train=y_fit)
                emb_test = extractor.transform(test_feats)
                logger.info(f"  Embedding dims: {emb_train.shape[1]}")

                emb_train_t = torch.from_numpy(emb_train)
                emb_test_t = torch.from_numpy(emb_test)
                labels_train_t = torch.from_numpy(raw_labels)
                labels_test_t = torch.from_numpy(test_labels)

                # Wrap as dict-style batches compatible with get_features()
                emb_train_ds = TensorDataset(emb_train_t, labels_train_t)
                emb_test_ds = TensorDataset(emb_test_t, labels_test_t)

                # Override data loaders with embedding-based datasets
                # Use a collate_fn that wraps tensors in the dict format expected
                # by get_features() and the rest of the pipeline.
                def _emb_collate(
                    batch: list[tuple[Tensor, Tensor]],
                ) -> tuple[dict[str, Tensor], Tensor]:
                    xs, ys = zip(*batch)
                    return {"cont_features": torch.stack(xs)}, torch.stack(ys)

                train_loader = DataLoader(
                    emb_train_ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=_emb_collate,
                )
                test_loader = DataLoader(
                    emb_test_ds,
                    batch_size=config.batch_size,
                    shuffle=False,
                    collate_fn=_emb_collate,
                )

                # MLPEncoder sees only continuous features from tree embeddings
                emb_n_continuous = emb_train.shape[1]
                emb_category_sizes: list[int] = []
            else:
                train_loader = DataLoader(
                    train_dataset, batch_size=config.batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=config.batch_size, shuffle=False
                )
                emb_n_continuous = metadata.n_continuous
                emb_category_sizes = metadata.category_sizes

            base_encoder = MLPEncoder(
                n_categories=emb_category_sizes,
                n_continuous=emb_n_continuous,
                dim=config.embedding_dim,
                hidden_dims=[config.hidden_dim, config.hidden_dim],
            ).to(device)
            base_encoder = pretrain_encoder(
                base_encoder, train_loader, metadata.n_classes, config, device
            )

            # Split methods into RQ-based and baselines
            rq_to_run = [m for m in methods if m in RQ_METHODS]
            baseline_to_run = [m for m in methods if m not in RQ_METHODS]

            # --- RQ methods: fit codebook ONCE, share across all ---
            if rq_to_run:
                # Determine the input space for RQ
                if config.feature_mode in ("raw", "selected"):
                    # Use raw/selected features for RQ codebook
                    cache = FeatureCache(config.feature_cache_dir)
                    features_np, labels_np = cache.get_or_compute_raw_features(
                        dataset_name,
                        lambda: collect_raw_features(train_dataset),
                    )
                    if config.feature_mode == "selected":
                        selected_indices, _ = cache.get_or_compute_selection(
                            dataset_name,
                            "unsupervised",
                            lambda: select_features_unsupervised(features_np),
                        )
                        features_np = features_np[:, selected_indices]
                        logger.info(
                            f"  Feature selection (unsupervised): "
                            f"{len(selected_indices)} features selected"
                        )
                    # Normalize before RQ
                    features_scaled, _, _ = normalize_for_rq(features_np)
                    all_embeddings_t = torch.from_numpy(features_scaled).to(device)

                    # Collect labels from dataset
                    collect_loader = DataLoader(
                        train_dataset, batch_size=config.batch_size, shuffle=False
                    )
                    all_labels_list: list[Tensor] = []
                    all_cont_list: list[Tensor] = []
                    all_cat_list: list[Tensor] = []
                    has_cat = False
                    with torch.no_grad():
                        for batch_features, batch_y in collect_loader:
                            cont_features, cat_features = get_features(
                                batch_features, device
                            )
                            all_labels_list.append(batch_y)
                            all_cont_list.append(cont_features.cpu())
                            if cat_features is not None:
                                all_cat_list.append(cat_features.cpu())
                                has_cat = True
                    all_labels_t = torch.cat(all_labels_list)
                    all_cont_t = torch.cat(all_cont_list)
                    all_cat_t = torch.cat(all_cat_list) if has_cat else None
                    rq_embedding_dim = features_scaled.shape[1]
                else:
                    # Embeddings mode: use encoder embeddings for RQ codebook
                    base_encoder.eval()
                    collect_loader = DataLoader(
                        train_dataset if config.embedding_type == "raw" else emb_train_ds,
                        batch_size=config.batch_size,
                        shuffle=False,
                        collate_fn=_emb_collate if config.embedding_type != "raw" else None,
                    )
                    all_embeddings_list: list[Tensor] = []
                    all_labels_list: list[Tensor] = []
                    all_cont_list: list[Tensor] = []
                    all_cat_list: list[Tensor] = []
                    has_cat = False
                    with torch.no_grad():
                        for batch_features, batch_y in collect_loader:
                            cont_features, cat_features = get_features(
                                batch_features, device
                            )
                            emb = base_encoder(
                                cat_features=cat_features, cont_features=cont_features
                            )
                            all_embeddings_list.append(emb.cpu())
                            all_labels_list.append(batch_y)
                            all_cont_list.append(cont_features.cpu())
                            if cat_features is not None:
                                all_cat_list.append(cat_features.cpu())
                                has_cat = True

                    all_embeddings_t = torch.cat(all_embeddings_list)
                    all_labels_t = torch.cat(all_labels_list)
                    all_cont_t = torch.cat(all_cont_list)
                    all_cat_t = torch.cat(all_cat_list) if has_cat else None
                    rq_embedding_dim = config.embedding_dim

                # Auto d_cut
                if config.auto_d_cut:
                    d_cut_used, _ = find_optimal_d_cut(
                        all_embeddings_t,
                        all_labels_t,
                        max_levels=config.n_levels,
                        nbits=config.nbits,
                        use_gpu=False,
                    )
                else:
                    d_cut_used = config.d_cut

                # Fit RQ ONCE (CPU to avoid faiss GPU memory leaks between seeds)
                faiss_config = FaissMRQConfig(
                    n_levels=config.n_levels,
                    nbits=config.nbits,
                    d_cut=d_cut_used,
                    use_gpu=False,
                    beam_size=config.beam_size,
                )
                mrq = FaissMRQModule(
                    n_classes=metadata.n_classes,
                    embedding_dim=rq_embedding_dim,
                    config=faiss_config,
                )
                codebook_start = time.time()
                mrq.fit(all_embeddings_t, all_labels_t)
                codebook_time = time.time() - codebook_start

                with torch.no_grad():
                    views = mrq.generate_views(all_embeddings_t)
                recon_error = mrq.get_reconstruction_error(
                    all_embeddings_t, all_labels_t
                )

                # Pre-compute prototypes and recon quality
                prototypes = _compute_prototypes(all_embeddings_t, all_labels_t)
                recon_quality = _compute_recon_quality(
                    all_embeddings_t, views, all_labels_t
                )

                # Run each RQ method from same starting encoder
                for method in rq_to_run:
                    logger.info(f"  Running: {method}, seed={seed}")
                    try:
                        method_start = time.time()
                        method_encoder = copy.deepcopy(base_encoder)
                        method_encoder, _ = train_contrastive(
                            method_encoder,
                            train_loader,
                            config,
                            device,
                            method,
                            views,
                            labels=all_labels_t,
                            mrq=mrq,
                            all_cont_features=all_cont_t,
                            all_cat_features=all_cat_t,
                            prototypes=prototypes,
                            recon_quality=recon_quality,
                        )
                        accuracy = linear_probe_eval(
                            method_encoder, train_loader, test_loader, device
                        )
                        result = SingleRunResult(
                            dataset=dataset_name,
                            seed=seed,
                            method=method,
                            accuracy=accuracy,
                            time_seconds=time.time() - method_start,
                            codebook_time_seconds=codebook_time,
                            d_cut=d_cut_used,
                            reconstruction_error=recon_error,
                        )
                        all_results.append(result)
                        logger.info(f"    Accuracy: {result.accuracy:.4f}")
                        print(
                            f"    {dataset_name} seed={seed} {method}: {result.accuracy:.4f} ({result.time_seconds:.0f}s)",
                            flush=True,
                        )
                    except Exception as e:
                        logger.error(f"    Failed: {e}")
                        print(
                            f"    {dataset_name} seed={seed} {method}: FAILED - {e}",
                            flush=True,
                        )
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Free RQ resources
                del mrq, views, all_embeddings_t
                del prototypes, recon_quality
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --- Baseline methods: no RQ needed ---
            for method in baseline_to_run:
                logger.info(f"  Running: {method}, seed={seed}")
                try:
                    method_start = time.time()
                    method_encoder = copy.deepcopy(base_encoder)
                    method_encoder, _ = train_contrastive(
                        method_encoder,
                        train_loader,
                        config,
                        device,
                        method,
                    )
                    accuracy = linear_probe_eval(
                        method_encoder, train_loader, test_loader, device
                    )
                    result = SingleRunResult(
                        dataset=dataset_name,
                        seed=seed,
                        method=method,
                        accuracy=accuracy,
                        time_seconds=time.time() - method_start,
                    )
                    all_results.append(result)
                    logger.info(f"    Accuracy: {result.accuracy:.4f}")
                    print(
                        f"    {dataset_name} seed={seed} {method}: {result.accuracy:.4f} ({result.time_seconds:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    print(
                        f"    {dataset_name} seed={seed} {method}: FAILED - {e}",
                        flush=True,
                    )
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Compute summary statistics
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_name in config.datasets:
        summary[dataset_name] = {}
        for method in methods:
            method_results = [
                r
                for r in all_results
                if r.dataset == dataset_name and r.method == method
            ]
            if method_results:
                accuracies = [r.accuracy for r in method_results]
                summary[dataset_name][method] = {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "min": float(np.min(accuracies)),
                    "max": float(np.max(accuracies)),
                    "n_runs": len(accuracies),
                }

    # Build output
    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_hash": get_git_hash(),
            "config": asdict(config),
            "device": str(device),
            "pytorch_version": torch.__version__,
        },
        "results": [asdict(r) for r in all_results],
        "summary": summary,
    }

    # Significance tests (paired t-test + bootstrap CI)
    if any(len(v) >= 2 for v in summary.values()):
        significance = _compute_per_dataset_significance(all_results, config.datasets)
        output["significance"] = significance

        # Log significance results
        if significance:
            logger.info("\n" + "=" * 80)
            logger.info("SIGNIFICANCE TESTS (best-ours vs best-baseline)")
            logger.info("=" * 80)
            for ds, sig in significance.items():
                logger.info(f"\n  Dataset: {ds}")
                log_significance(sig, logger, metric_name="accuracy")

    return output


def _compute_per_dataset_significance(
    all_results: list[SingleRunResult],
    datasets: list[str],
) -> dict[str, dict[str, object]]:
    """Compute pairwise significance for each dataset using shared utility.

    Groups results by dataset, then delegates to compute_pairwise_significance.

    Returns:
        Dict mapping dataset -> significance results.
    """
    significance: dict[str, dict[str, object]] = {}

    for dataset_name in datasets:
        ds_results = [r for r in all_results if r.dataset == dataset_name]
        if not ds_results:
            continue

        # Group accuracies by method, with seed-ordered lists for pairing
        method_seed_acc: dict[str, dict[int, float]] = {}
        for r in ds_results:
            method_seed_acc.setdefault(r.method, {})[r.seed] = r.accuracy

        # Find common seeds across all methods for pairing
        all_seed_sets = [set(v.keys()) for v in method_seed_acc.values()]
        if not all_seed_sets:
            continue
        common_seeds = sorted(set.intersection(*all_seed_sets))
        if len(common_seeds) < 2:
            continue

        # Build paired lists (same seed order for all methods)
        method_seed_values: dict[str, list[float]] = {}
        for method, seed_acc in method_seed_acc.items():
            method_seed_values[method] = [seed_acc[s] for s in common_seeds]

        sig = compute_pairwise_significance(
            method_seed_values,
            ours_methods=RQ_METHODS,
            higher_is_better=True,
        )
        if sig:
            significance[dataset_name] = sig

    return significance


def print_summary_table(output: dict[str, Any]) -> None:
    """Print a formatted summary table for the paper."""
    summary = output["summary"]

    # Collect all methods that actually have results
    all_methods: set[str] = set()
    for ds_data in summary.values():
        all_methods.update(ds_data.keys())
    methods = sorted(all_methods)

    logger.info("\n" + "=" * 100)
    logger.info("PAPER TABLE: RQ-based (ours) vs Baselines (Accuracy %)")
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
                row += f"{mean:>14.2f} "
            else:
                row += f"{'N/A':>15}"
        logger.info(row)

    logger.info("-" * 100)
    logger.info("* = ours (RQ-based), others = baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-dataset m-RQ experiments")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "bank_marketing", "german_credit"],
        help="Datasets to run experiments on",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/mrq_paper_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=8,
        help="Number of RQ levels",
    )
    parser.add_argument(
        "--d-cut",
        type=int,
        default=4,
        help="Trunk-tail boundary",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=6,
        help="Bits per level",
    )
    parser.add_argument(
        "--auto-d-cut",
        action="store_true",
        help="Auto-detect optimal d_cut per dataset",
    )
    parser.add_argument(
        "--contrastive-epochs",
        type=int,
        default=50,
        help="Contrastive training epochs",
    )
    parser.add_argument(
        "--view-refresh-interval",
        type=int,
        default=25,
        help="Refresh m-RQ views every N contrastive epochs (0=disable)",
    )
    parser.add_argument(
        "--mrq-noise-scale",
        type=float,
        default=0.1,
        help="Noise scale for m-RQ view augmentation",
    )
    parser.add_argument(
        "--encoder-epochs",
        type=int,
        default=20,
        help="Encoder pre-training epochs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with 1 seed and fewer epochs",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Include RQ-based supervised contrastive methods (RQ-SupCon+Proto, RQ-WeightedSupCon, RQ-HardNegSupCon, RQ-Proto+Weighted)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Run only these methods (e.g. --methods m-RQ Gaussian RQ-SupCon+Proto)",
    )
    parser.add_argument(
        "--prototype-weight",
        type=float,
        default=0.5,
        help="Prototype attraction weight for SupCon+Proto",
    )
    parser.add_argument(
        "--hard-neg-strength",
        type=float,
        default=0.5,
        help="Hard negative up-weighting strength for HardNegSupCon",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="embeddings",
        choices=["embeddings", "raw", "selected"],
        help="Input space for RQ: encoder embeddings, raw features, or selected features",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="results/feature_cache",
        help="Directory for caching raw features and feature selection results",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="raw",
        choices=["raw", "lgbm", "iforest"],
        help="Embedding type: raw features, LightGBM leaves, or IsolationForest leaves",
    )
    args = parser.parse_args()

    # Build config
    if args.quick:
        config = ExperimentConfig(
            datasets=args.datasets,
            seeds=[42],
            n_levels=args.n_levels,
            d_cut=args.d_cut,
            nbits=args.nbits,
            auto_d_cut=args.auto_d_cut,
            contrastive_epochs=10,
            encoder_epochs=5,
            view_refresh_interval=0,  # No refresh for quick runs
            include_hybrid_methods=args.include_hybrid,
            methods_filter=args.methods,
            prototype_weight=args.prototype_weight,
            hard_neg_strength=args.hard_neg_strength,
            feature_mode=args.feature_mode,
            feature_cache_dir=args.feature_cache_dir,
            embedding_type=args.embedding_type,
        )
    else:
        config = ExperimentConfig(
            datasets=args.datasets,
            seeds=args.seeds,
            n_levels=args.n_levels,
            d_cut=args.d_cut,
            nbits=args.nbits,
            auto_d_cut=args.auto_d_cut,
            contrastive_epochs=args.contrastive_epochs,
            view_refresh_interval=args.view_refresh_interval,
            mrq_noise_scale=args.mrq_noise_scale,
            encoder_epochs=args.encoder_epochs,
            include_hybrid_methods=args.include_hybrid,
            methods_filter=args.methods,
            prototype_weight=args.prototype_weight,
            hard_neg_strength=args.hard_neg_strength,
            feature_mode=args.feature_mode,
            feature_cache_dir=args.feature_cache_dir,
            embedding_type=args.embedding_type,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-Dataset m-RQ Experiments")
    logger.info("=" * 60)
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Seeds: {config.seeds}")
    logger.info(
        f"n_levels: {config.n_levels}, d_cut: {config.d_cut}, nbits: {config.nbits}"
    )
    logger.info(f"auto_d_cut: {config.auto_d_cut}")
    logger.info(f"contrastive_epochs: {config.contrastive_epochs}")

    # Run experiments
    output = run_all_experiments(config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mrq_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_file}")

    # Print summary table
    print_summary_table(output)


if __name__ == "__main__":
    main()
