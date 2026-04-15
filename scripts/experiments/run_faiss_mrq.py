#!/usr/bin/env python3


"""Faiss m-RQ benchmark: Compare sklearn k-means vs faiss RQ."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import logging
from modules.contrastive.faiss_mrq_module import (
    compare_sklearn_vs_faiss,
    FaissMRQConfig,
    FaissMRQModule,
    find_optimal_d_cut,
)
from modules.contrastive.losses import info_nce_loss
from modules.contrastive.projection import ProjectionHead
from modules.data import load_adult
from modules.encoders.tab_transformer import MLPEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class FaissTrainingConfig:
    """Training configuration."""

    # Faiss RQ params
    n_levels: int = 8
    nbits: int = 6
    d_cut: int = 4
    beam_size: int = 5
    use_gpu: bool = True

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

    # Projection head
    projection_dim: int = 32


@dataclass
class FaissResult:
    """Result from faiss m-RQ experiment."""

    method: str
    linear_probe_accuracy: float
    total_time_seconds: float
    codebook_time_seconds: float
    trunk_time_seconds: float
    tail_time_seconds: float
    contrastive_time_seconds: float
    reconstruction_error: float | None = None
    auto_d_cut: int | None = None


def get_features(
    batch_features: dict[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Extract continuous and categorical features from batch."""
    cont_features = batch_features["cont_features"].to(device)
    cat_features = batch_features["cat_features"].to(device)
    return cont_features, cat_features


def pretrain_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,  # pyre-ignore[11]
    n_classes: int,
    config: FaissTrainingConfig,
    device: torch.device,
    metadata: object,  # pyre-ignore[2]
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
    for epoch in range(config.encoder_epochs):
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
            logger.info(f"  Encoder pre-train epoch {epoch + 1}: loss={avg_loss:.4f}")

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


def collect_embeddings(
    encoder: nn.Module,
    train_loader: DataLoader,  # pyre-ignore[11]
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


def train_faiss_mrq_contrastive(
    encoder: nn.Module,
    precomputed_loader: DataLoader,  # pyre-ignore[11]
    config: FaissTrainingConfig,
    device: torch.device,
    mrq: FaissMRQModule | None = None,
    train_loader: DataLoader | None = None,  # pyre-ignore[11]
) -> tuple[nn.Module, float]:
    """Train encoder with faiss m-RQ contrastive loss using pre-computed views."""
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
    view_refresh_interval = 10

    for epoch in range(config.contrastive_epochs):
        # Periodically refresh views as the encoder changes
        if (
            view_refresh_interval > 0
            and epoch > 0
            and epoch % view_refresh_interval == 0
            and mrq is not None
            and train_loader is not None
        ):
            encoder.eval()
            with torch.no_grad():
                new_embeddings, new_labels = collect_embeddings(
                    encoder, train_loader, device
                )
                mrq.fit(new_embeddings, new_labels)
                new_views = mrq.generate_views(new_embeddings)
            precomputed_dataset = TensorDataset(new_embeddings, new_views, new_labels)
            precomputed_loader = DataLoader(
                precomputed_dataset, batch_size=config.batch_size, shuffle=True
            )
            # Free GPU memory from old faiss RQ objects
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            encoder.train()

        total_loss = 0.0
        for batch_embeddings, batch_views, batch_labels in precomputed_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_views = batch_views.to(device)
            batch_labels = batch_labels.to(device)

            loss = mrq_contrastive_loss(
                batch_embeddings,
                batch_views,
                batch_labels,
                projection_head,
                config.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(precomputed_loader)
            logger.info(
                f"  Faiss m-RQ contrastive epoch {epoch + 1}: loss={avg_loss:.4f}"
            )

    contrastive_time = time.time() - start_time
    return encoder, contrastive_time


def linear_probe_eval(
    encoder: nn.Module,
    train_loader: DataLoader,  # pyre-ignore[11]
    test_loader: DataLoader,  # pyre-ignore[11]
    device: torch.device,
    metadata: object,  # pyre-ignore[2]
) -> float:
    """Evaluate encoder using linear probe."""
    encoder.eval()

    # Collect train embeddings
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

    # Collect test embeddings
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

    # Train and evaluate linear probe
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return float(accuracy_score(y_test, y_pred))


def run_faiss_mrq_experiment(
    config: FaissTrainingConfig,
    auto_d_cut: bool = False,
) -> FaissResult:
    """Run faiss m-RQ experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading Adult dataset...")
    train_dataset, val_dataset, test_dataset, metadata = load_adult()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    n_classes = metadata.n_classes
    logger.info(
        f"Loaded: {metadata.n_continuous} continuous, "
        f"{metadata.n_categorical} categorical, {n_classes} classes"
    )

    total_start = time.time()

    # Create and pre-train encoder
    encoder = MLPEncoder(
        n_categories=metadata.category_sizes,
        n_continuous=metadata.n_continuous,
        dim=config.embedding_dim,
        hidden_dims=[config.hidden_dim, config.hidden_dim],
    ).to(device)

    encoder = pretrain_encoder(
        encoder, train_loader, n_classes, config, device, metadata
    )

    # Collect embeddings
    encoder.eval()
    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []
    with torch.no_grad():
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_embeddings.append(emb)
            all_labels.append(batch_y.to(device))
    all_embeddings_t = torch.cat(all_embeddings)
    all_labels_t = torch.cat(all_labels)

    # Auto d_cut if requested (uses CPU to avoid GPU memory fragmentation)
    detected_d_cut = None
    if auto_d_cut:
        logger.info("Auto-detecting optimal d_cut...")
        detected_d_cut, divergences = find_optimal_d_cut(
            all_embeddings_t,
            all_labels_t,
            max_levels=config.n_levels,
            nbits=config.nbits,
            use_gpu=False,  # Force CPU for d_cut detection (GPU mem not released)
        )
        logger.info(f"Auto-detected d_cut: {detected_d_cut}")
        d_cut_to_use = detected_d_cut
    else:
        d_cut_to_use = config.d_cut

    # Create and fit faiss m-RQ
    logger.info(
        f"Fitting faiss m-RQ (n_levels={config.n_levels}, d_cut={d_cut_to_use})..."
    )
    faiss_config = FaissMRQConfig(
        n_levels=config.n_levels,
        nbits=config.nbits,
        d_cut=d_cut_to_use,
        use_gpu=config.use_gpu,
        beam_size=config.beam_size,
    )
    mrq = FaissMRQModule(
        n_classes=n_classes,
        embedding_dim=config.embedding_dim,
        config=faiss_config,
    )

    codebook_start = time.time()
    mrq.fit(all_embeddings_t, all_labels_t)
    codebook_time = time.time() - codebook_start

    # Generate views
    logger.info("Generating views...")
    with torch.no_grad():
        all_views = mrq.generate_views(all_embeddings_t)

    # Reconstruction error
    recon_error = mrq.get_reconstruction_error(all_embeddings_t, all_labels_t)
    logger.info(f"Reconstruction error: {recon_error:.4f}")

    # Contrastive training
    logger.info("Contrastive training with faiss m-RQ views...")
    precomputed_dataset = TensorDataset(all_embeddings_t, all_views, all_labels_t)
    precomputed_loader = DataLoader(
        precomputed_dataset, batch_size=config.batch_size, shuffle=True
    )
    encoder, contrastive_time = train_faiss_mrq_contrastive(
        encoder, precomputed_loader, config, device, mrq=mrq, train_loader=train_loader
    )

    # Evaluate
    accuracy = linear_probe_eval(encoder, train_loader, test_loader, device, metadata)
    total_time = time.time() - total_start

    logger.info("Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(
        f"  Codebook time: {codebook_time:.2f}s (trunk={mrq.trunk_fit_time:.2f}s, tails={mrq.tail_fit_time:.2f}s)"
    )
    logger.info(f"  Contrastive time: {contrastive_time:.1f}s")

    return FaissResult(
        method=f"faiss_mrq_L{config.n_levels}_D{d_cut_to_use}",
        linear_probe_accuracy=accuracy,
        total_time_seconds=total_time,
        codebook_time_seconds=codebook_time,
        trunk_time_seconds=mrq.trunk_fit_time,
        tail_time_seconds=mrq.tail_fit_time,
        contrastive_time_seconds=contrastive_time,
        reconstruction_error=recon_error,
        auto_d_cut=detected_d_cut,
    )


def run_sklearn_vs_faiss_benchmark(config: FaissTrainingConfig) -> dict[str, float]:
    """Run head-to-head benchmark of sklearn vs faiss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading Adult dataset...")
    train_dataset, _val_dataset, _test_dataset, metadata = load_adult()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    n_classes = metadata.n_classes

    # Create and pre-train encoder
    encoder = MLPEncoder(
        n_categories=metadata.category_sizes,
        n_continuous=metadata.n_continuous,
        dim=config.embedding_dim,
        hidden_dims=[config.hidden_dim, config.hidden_dim],
    ).to(device)

    encoder = pretrain_encoder(
        encoder, train_loader, n_classes, config, device, metadata
    )

    # Collect embeddings
    encoder.eval()
    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []
    with torch.no_grad():
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_embeddings.append(emb)
            all_labels.append(batch_y.to(device))
    all_embeddings_t = torch.cat(all_embeddings)
    all_labels_t = torch.cat(all_labels)

    # Run benchmark
    logger.info("\n" + "=" * 60)
    logger.info("SKLEARN vs FAISS BENCHMARK")
    logger.info("=" * 60)

    results = compare_sklearn_vs_faiss(
        all_embeddings_t,
        all_labels_t,
        n_levels=config.n_levels,
        nbits=config.nbits,
        d_cut=config.d_cut,
    )

    logger.info("\nRESULTS:")
    logger.info(f"  faiss RQ: {results['faiss_time']:.2f}s")
    logger.info(f"    - trunk: {results['faiss_trunk_time']:.2f}s")
    logger.info(f"    - tails: {results['faiss_tail_time']:.2f}s")

    return results


def run_baseline_comparison(
    config: FaissTrainingConfig,
) -> list[dict[str, float | str]]:
    """Run comprehensive m-RQ vs baseline comparison.

    Baselines:
    - Gaussian: Random Gaussian noise augmentation
    - Dropout: Feature dropout augmentation
    - SCARF: Random feature corruption with noise replacement
    - SupCon: Supervised contrastive (uses label to form positives)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading Adult dataset...")
    train_dataset, val_dataset, test_dataset, metadata = load_adult()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    n_classes = metadata.n_classes
    logger.info(
        f"Loaded: {metadata.n_continuous} continuous, "
        f"{metadata.n_categorical} categorical, {n_classes} classes"
    )

    results: list[dict[str, float | str]] = []

    # Helper to create fresh encoder
    def create_encoder() -> MLPEncoder:
        return MLPEncoder(
            n_categories=metadata.category_sizes,
            n_continuous=metadata.n_continuous,
            dim=config.embedding_dim,
            hidden_dims=[config.hidden_dim, config.hidden_dim],
        ).to(device)

    # =====================================================================
    # 1. Faiss m-RQ (our method)
    # =====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Method 1: Faiss m-RQ (ours)")
    logger.info("=" * 60)

    encoder = create_encoder()
    encoder = pretrain_encoder(
        encoder, train_loader, n_classes, config, device, metadata
    )

    # Collect embeddings
    encoder.eval()
    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []
    with torch.no_grad():
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_embeddings.append(emb)
            all_labels.append(batch_y.to(device))
    all_embeddings_t = torch.cat(all_embeddings)
    all_labels_t = torch.cat(all_labels)

    # Fit faiss m-RQ
    faiss_config = FaissMRQConfig(
        n_levels=config.n_levels,
        nbits=config.nbits,
        d_cut=config.d_cut,
        use_gpu=config.use_gpu,
        beam_size=config.beam_size,
    )
    mrq = FaissMRQModule(
        n_classes=n_classes,
        embedding_dim=config.embedding_dim,
        config=faiss_config,
    )

    mrq_start = time.time()
    mrq.fit(all_embeddings_t, all_labels_t)
    codebook_time = mrq.trunk_fit_time + mrq.tail_fit_time

    # Generate views and train
    with torch.no_grad():
        all_views = mrq.generate_views(all_embeddings_t)

    precomputed_dataset = TensorDataset(all_embeddings_t, all_views, all_labels_t)
    precomputed_loader = DataLoader(
        precomputed_dataset, batch_size=config.batch_size, shuffle=True
    )
    encoder, contrastive_time = train_faiss_mrq_contrastive(
        encoder, precomputed_loader, config, device, mrq=mrq, train_loader=train_loader
    )
    mrq_total_time = time.time() - mrq_start

    mrq_accuracy = linear_probe_eval(
        encoder, train_loader, test_loader, device, metadata
    )
    logger.info(f"  m-RQ Accuracy: {mrq_accuracy:.4f}, Time: {mrq_total_time:.1f}s")

    results.append(
        {
            "method": "faiss_m-RQ",
            "accuracy": mrq_accuracy,
            "time": mrq_total_time,
            "codebook_time": codebook_time,
        }
    )

    # =====================================================================
    # 2-5. Baseline methods (Gaussian, Dropout, SCARF, SupCon)
    # =====================================================================
    baseline_methods = [
        ("Gaussian", 0.1, 0.0),  # (name, noise_scale, dropout_rate)
        ("Dropout", 0.0, 0.3),
        ("SCARF", 0.1, 0.3),
        ("SupCon", 0.0, 0.0),  # SupCon uses label-based positives
    ]

    for method_name, noise_scale, dropout_rate in baseline_methods:
        logger.info("\n" + "=" * 60)
        logger.info(f"Method: {method_name}")
        logger.info("=" * 60)

        encoder = create_encoder()
        encoder = pretrain_encoder(
            encoder, train_loader, n_classes, config, device, metadata
        )

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

        method_start = time.time()
        for epoch in range(config.contrastive_epochs):
            total_loss = 0.0
            for batch_features, batch_y in train_loader:
                cont_features, cat_features = get_features(batch_features, device)
                batch_y = batch_y.to(device)

                embeddings = encoder(
                    cat_features=cat_features, cont_features=cont_features
                )

                if method_name == "Gaussian":
                    noise = torch.randn_like(embeddings) * noise_scale
                    aug_embeddings = embeddings + noise
                elif method_name == "Dropout":
                    dropout = nn.Dropout(dropout_rate)
                    aug_embeddings = dropout(embeddings)
                elif method_name == "SCARF":
                    mask = torch.rand_like(embeddings) < dropout_rate
                    noise = torch.randn_like(embeddings) * noise_scale
                    aug_embeddings = torch.where(mask, noise, embeddings)
                elif method_name == "SupCon":
                    # SupCon: positives are samples with same label
                    z = projection_head(embeddings)
                    z = nn.functional.normalize(z, dim=1)
                    # Compute SupCon loss
                    labels = batch_y.unsqueeze(0)
                    mask_pos = (labels == labels.T).float()
                    mask_pos.fill_diagonal_(0)

                    sim = torch.matmul(z, z.T) / config.temperature
                    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
                    sim = sim - sim_max.detach()

                    exp_sim = torch.exp(sim)
                    mask_self = torch.eye(len(z), device=device)
                    denom = (exp_sim * (1 - mask_self)).sum(dim=1, keepdim=True)

                    log_prob = sim - torch.log(denom + 1e-8)
                    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (
                        mask_pos.sum(dim=1) + 1e-8
                    )
                    loss = -mean_log_prob_pos.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    continue

                # For non-SupCon methods: standard InfoNCE
                z = projection_head(embeddings)
                z_aug = projection_head(aug_embeddings)
                loss = info_nce_loss(z, z_aug, config.temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f"  {method_name} epoch {epoch + 1}: loss={avg_loss:.4f}")

        method_time = time.time() - method_start
        accuracy = linear_probe_eval(
            encoder, train_loader, test_loader, device, metadata
        )
        logger.info(
            f"  {method_name} Accuracy: {accuracy:.4f}, Time: {method_time:.1f}s"
        )

        results.append(
            {
                "method": method_name,
                "accuracy": accuracy,
                "time": method_time,
                "codebook_time": 0.0,
            }
        )

    # =====================================================================
    # Summary
    # =====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Method':<15} {'Accuracy':>10} {'Time (s)':>10} {'Δ vs m-RQ':>12}")
    logger.info("-" * 50)

    mrq_acc = results[0]["accuracy"]
    for r in sorted(results, key=lambda x: -x["accuracy"]):  # pyre-ignore[6]
        delta = r["accuracy"] - mrq_acc  # pyre-ignore[58]
        delta_str = f"{delta:+.4f}" if r["method"] != "faiss_m-RQ" else "—"
        logger.info(
            f"{r['method']:<15} {r['accuracy']:>10.4f} {r['time']:>10.1f} {delta_str:>12}"
        )

    # Declare winner
    best = max(results, key=lambda x: x["accuracy"])  # pyre-ignore[6]
    logger.info("")
    if best["method"] == "faiss_m-RQ":
        second_best = max(
            [r for r in results if r["method"] != "faiss_m-RQ"],
            key=lambda x: x["accuracy"],  # pyre-ignore[6]
        )
        margin = mrq_acc - second_best["accuracy"]  # pyre-ignore[58]
        logger.info(f"✅ m-RQ WINS by {margin:.4f} over {second_best['method']}")
    else:
        margin = best["accuracy"] - mrq_acc  # pyre-ignore[58]
        logger.info(f"❌ {best['method']} beats m-RQ by {margin:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Faiss m-RQ experiment")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/faiss_mrq_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--contrastive-epochs",
        type=int,
        default=50,
        help="Number of contrastive training epochs",
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
        help="Trunk-tail boundary (trunk levels)",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=6,
        help="Bits per level (K = 2^nbits)",
    )
    parser.add_argument(
        "--auto-d-cut",
        action="store_true",
        help="Automatically detect optimal d_cut",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run sklearn vs faiss benchmark only",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run full m-RQ vs baselines comparison",
    )
    args = parser.parse_args()

    config = FaissTrainingConfig(
        n_levels=args.n_levels,
        d_cut=args.d_cut,
        nbits=args.nbits,
        contrastive_epochs=args.contrastive_epochs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Faiss m-RQ Experiment")
    logger.info("=" * 60)
    logger.info("Config:")
    logger.info(f"  n_levels: {config.n_levels}")
    logger.info(f"  d_cut: {config.d_cut}")
    logger.info(f"  nbits: {config.nbits} (K={2**config.nbits})")
    logger.info(f"  contrastive_epochs: {config.contrastive_epochs}")
    logger.info(f"  auto_d_cut: {args.auto_d_cut}")

    if args.benchmark:
        results = run_sklearn_vs_faiss_benchmark(config)
        output_file = output_dir / "sklearn_vs_faiss_benchmark.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=float)
    elif args.compare:
        results = run_baseline_comparison(config)
        output_file = output_dir / "mrq_vs_baselines.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=float)
    else:
        result = run_faiss_mrq_experiment(config, auto_d_cut=args.auto_d_cut)
        output_file = output_dir / "faiss_mrq_result.json"
        with open(output_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
