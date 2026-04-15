#!/usr/bin/env python3


"""Compression-accuracy tradeoff experiment on real tabular datasets.

Demonstrates that RQ trunk codes achieve near-full classification accuracy
at extreme compression ratios, directly validating the claim that trunk
codes are task-conditioned sufficient statistics for Y.

Representations compared (at matched or near-matched bitrates):
    1. Raw embedding: logistic regression on full d-dimensional embedding
    2. RQ trunk codes (d_cut=1..max_depth): logistic regression on one-hot
       encoded trunk codes (or centroid reconstructions)
    3. KMeans cluster IDs (at matched #clusters): same protocol
    4. PCA projections (at matched dimensionality): logistic regression
    5. Random projections: Gaussian random matrix to matched dimensions
    6. RQ reconstructions: logistic regression on decoded trunk vectors
    7. KMeans reconstructions: logistic regression on centroid vectors

Key metrics:
    - Classification accuracy vs bits per sample
    - Compression ratio vs accuracy loss (relative to full embedding)
    - Label purity (V-measure, ARI) of discrete codes vs KMeans

Example:
    python3 scripts/experiments/run_compression_experiment.py -- \\
        --dataset adult --n-encoder-seeds 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import logging
from modules.data import DATASET_REGISTRY, load_dataset
from modules.encoders.tab_transformer import MLPEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, classification_report, v_measure_score
from sklearn.random_projection import GaussianRandomProjection
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
class CompressionConfig:
    """Configuration for compression-accuracy tradeoff experiment."""

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
    nbits: int = 6
    max_depth: int = 4  # Evaluate d_cut=1..max_depth

    device: str = "cuda"


# =============================================================================
# Representation builders
# =============================================================================


def _fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
) -> dict[str, float]:
    """Fit logistic regression and return accuracy metrics."""
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
    )
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    return {"train_acc": train_acc, "test_acc": test_acc}


def _codes_to_onehot(
    codes_train: np.ndarray,
    codes_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert code arrays to one-hot feature matrices.

    Handles multi-column codes by concatenating one-hot per column.
    """
    if codes_train.ndim == 1:
        codes_train = codes_train.reshape(-1, 1)
        codes_test = codes_test.reshape(-1, 1)

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for col in range(codes_train.shape[1]):
        col_train = codes_train[:, col]
        col_test = codes_test[:, col]
        # Find all unique values across train and test
        all_vals = np.unique(np.concatenate([col_train, col_test]))
        val_to_idx = {int(v): i for i, v in enumerate(all_vals)}
        n_vals = len(all_vals)

        oh_train = np.zeros((len(col_train), n_vals), dtype=np.float32)
        for i, v in enumerate(col_train):
            oh_train[i, val_to_idx[int(v)]] = 1.0

        oh_test = np.zeros((len(col_test), n_vals), dtype=np.float32)
        for i, v in enumerate(col_test):
            idx = val_to_idx.get(int(v))
            if idx is not None:
                oh_test[i, idx] = 1.0

        train_parts.append(oh_train)
        test_parts.append(oh_test)

    return np.hstack(train_parts), np.hstack(test_parts)


def evaluate_representations(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    nbits: int,
    max_depth: int,
    encoder_seed: int,
) -> dict[str, dict[str, object]]:
    """Evaluate classification accuracy of all representations.

    Returns:
        representation_name -> {
            "test_acc": float,
            "train_acc": float,
            "bits_per_sample": float,
            "n_features": int,
            "compression_ratio": float,
            ...
        }
    """
    import faiss

    dim = train_emb.shape[1]
    raw_bits = dim * 32  # float32

    results: dict[str, dict[str, object]] = {}

    # 1. Raw embedding (baseline)
    logger.info("    Evaluating: Raw embedding...")
    raw_result = _fit_logreg(train_emb, train_labels, test_emb, test_labels)
    results["Raw-Embedding"] = {
        **raw_result,
        "bits_per_sample": float(raw_bits),
        "n_features": dim,
        "compression_ratio": 1.0,
    }
    raw_test_acc = raw_result["test_acc"]

    # 2-3. RQ trunk codes and reconstructions at each d_cut
    for d_cut in range(1, max_depth + 1):
        bits = d_cut * nbits
        logger.info(f"    Evaluating: RQ d_cut={d_cut} ({bits} bits)...")

        rq = faiss.ResidualQuantizer(dim, d_cut, nbits)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.verbose = False
        rq.train(train_emb)

        codes_train = rq.compute_codes(train_emb).astype(np.int64)
        codes_test = rq.compute_codes(test_emb).astype(np.int64)

        # RQ codes → one-hot → logistic regression
        oh_train, oh_test = _codes_to_onehot(codes_train, codes_test)
        code_result = _fit_logreg(oh_train, train_labels, oh_test, test_labels)

        # Label purity metrics
        # Convert multi-column codes to single cluster labels for V-measure/ARI
        code_keys_train = [tuple(c) for c in codes_train]
        unique_keys = list(set(code_keys_train))
        key_to_id = {k: i for i, k in enumerate(unique_keys)}
        cluster_train = np.array([key_to_id[k] for k in code_keys_train])
        code_keys_test = [tuple(c) for c in codes_test]
        cluster_test = np.array([key_to_id.get(k, -1) for k in code_keys_test])

        v_measure = float(v_measure_score(train_labels, cluster_train))
        ari = float(adjusted_rand_score(train_labels, cluster_train))
        h_y_given_z = conditional_entropy(train_labels, codes_train)
        n_unique = len(unique_keys)

        results[f"RQ-Codes-d{d_cut}"] = {
            **code_result,
            "bits_per_sample": float(bits),
            "n_features": int(oh_train.shape[1]),
            "compression_ratio": raw_bits / bits,
            "n_unique_codes": n_unique,
            "v_measure": v_measure,
            "ari": ari,
            "h_y_given_z": h_y_given_z,
        }

        # RQ reconstructions → logistic regression
        recon_train: np.ndarray = rq.decode(rq.compute_codes(train_emb))
        recon_test: np.ndarray = rq.decode(rq.compute_codes(test_emb))
        recon_result = _fit_logreg(recon_train, train_labels, recon_test, test_labels)
        results[f"RQ-Recon-d{d_cut}"] = {
            **recon_result,
            "bits_per_sample": float(bits),
            "n_features": dim,
            "compression_ratio": raw_bits / bits,
        }

    # 4-5. KMeans at matched cluster counts
    for d_cut in range(1, max_depth + 1):
        n_clusters = min(2 ** (d_cut * nbits), train_emb.shape[0])
        # Cap at reasonable size for one-hot encoding
        n_clusters = min(n_clusters, 10000)
        bits = math.log2(n_clusters) if n_clusters > 1 else 1
        logger.info(f"    Evaluating: KMeans k={n_clusters} ({bits:.0f} bits)...")

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=encoder_seed,
            n_init=3,
            batch_size=1024,
        )
        km.fit(train_emb)
        km_train = km.labels_
        km_test = km.predict(test_emb)

        # KMeans IDs → one-hot → logistic regression
        oh_train, oh_test = _codes_to_onehot(
            km_train.reshape(-1, 1), km_test.reshape(-1, 1)
        )
        km_code_result = _fit_logreg(oh_train, train_labels, oh_test, test_labels)

        v_measure = float(v_measure_score(train_labels, km_train))
        ari = float(adjusted_rand_score(train_labels, km_train))
        h_y_given_z = conditional_entropy(train_labels, km_train.reshape(-1, 1))

        results[f"KM-Codes-k{n_clusters}"] = {
            **km_code_result,
            "bits_per_sample": float(bits),
            "n_features": int(oh_train.shape[1]),
            "compression_ratio": raw_bits / bits,
            "n_unique_codes": int(n_clusters),
            "v_measure": v_measure,
            "ari": ari,
            "h_y_given_z": h_y_given_z,
        }

        # KMeans reconstructions → logistic regression
        km_recon_train = km.cluster_centers_[km_train]
        km_recon_test = km.cluster_centers_[km_test]
        km_recon_result = _fit_logreg(
            km_recon_train, train_labels, km_recon_test, test_labels
        )
        results[f"KM-Recon-k{n_clusters}"] = {
            **km_recon_result,
            "bits_per_sample": float(bits),
            "n_features": dim,
            "compression_ratio": raw_bits / bits,
        }

    # 6. PCA at matched dimensions
    for n_comp in [1, 2, 4, 8, 16, 32]:
        if n_comp >= dim:
            continue
        bits = n_comp * 32
        logger.info(f"    Evaluating: PCA k={n_comp} ({bits} bits)...")
        pca = PCA(n_components=n_comp, random_state=encoder_seed)
        pca_train = pca.fit_transform(train_emb)
        pca_test = pca.transform(test_emb)
        explained_var = float(pca.explained_variance_ratio_.sum())

        pca_result = _fit_logreg(pca_train, train_labels, pca_test, test_labels)
        results[f"PCA-k{n_comp}"] = {
            **pca_result,
            "bits_per_sample": float(bits),
            "n_features": n_comp,
            "compression_ratio": raw_bits / bits,
            "explained_variance": explained_var,
        }

    # 7. Random projection at matched dimensions
    for n_comp in [1, 2, 4, 8, 16, 32]:
        if n_comp >= dim:
            continue
        bits = n_comp * 32
        logger.info(f"    Evaluating: Random projection k={n_comp}...")
        rp = GaussianRandomProjection(n_components=n_comp, random_state=encoder_seed)
        rp_train = rp.fit_transform(train_emb)
        rp_test = rp.transform(test_emb)

        rp_result = _fit_logreg(rp_train, train_labels, rp_test, test_labels)
        results[f"RandProj-k{n_comp}"] = {
            **rp_result,
            "bits_per_sample": float(bits),
            "n_features": n_comp,
            "compression_ratio": raw_bits / bits,
        }

    return results


# =============================================================================
# Main experiment
# =============================================================================


def run_compression_experiment(config: CompressionConfig) -> dict[str, object]:
    """End-to-end compression-accuracy tradeoff experiment."""
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
        logger.info(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")

        train_emb_np = train_emb.detach().numpy().astype(np.float32)
        test_emb_np = test_emb.detach().numpy().astype(np.float32)
        train_labels_np = train_labels.numpy()
        test_labels_np = test_labels.numpy()

        # Evaluate all representations
        logger.info("  Evaluating representations...")
        repr_results = evaluate_representations(
            train_emb_np,
            test_emb_np,
            train_labels_np,
            test_labels_np,
            config.nbits,
            config.max_depth,
            encoder_seed,
        )

        # Log results for this seed
        logger.info(f"\n  Results (seed={encoder_seed}):")
        header = (
            f"  {'Representation':<25s}  {'Bits':>6s}  {'Ratio':>7s}  "
            f"{'TrainAcc':>8s}  {'TestAcc':>8s}"
        )
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))

        # Sort by bits per sample for clean display
        sorted_names = sorted(
            repr_results.keys(),
            key=lambda k: repr_results[k]["bits_per_sample"],
        )
        for name in sorted_names:
            r = repr_results[name]
            logger.info(
                f"  {name:<25s}  {r['bits_per_sample']:>6.0f}  "
                f"{r['compression_ratio']:>7.1f}x  "
                f"{r['train_acc']:>8.4f}  {r['test_acc']:>8.4f}"
            )

        # Log label purity for discrete codes
        logger.info(f"\n  Label purity (discrete codes):")
        purity_header = (
            f"  {'Codes':<25s}  {'V-measure':>9s}  {'ARI':>8s}  "
            f"{'H(Y|Z)':>8s}  {'#Unique':>7s}"
        )
        logger.info(purity_header)
        logger.info("  " + "-" * (len(purity_header) - 2))
        for name in sorted_names:
            r = repr_results[name]
            if "v_measure" in r:
                logger.info(
                    f"  {name:<25s}  {r['v_measure']:>9.4f}  "
                    f"{r['ari']:>8.4f}  {r['h_y_given_z']:>8.4f}  "
                    f"{r['n_unique_codes']:>7d}"
                )

        seed_result: dict[str, object] = {
            "encoder_seed": encoder_seed,
            "n_train": int(train_emb_np.shape[0]),
            "n_test": int(test_emb_np.shape[0]),
            "representations": repr_results,
        }
        per_seed_results.append(seed_result)

    # ----- Aggregate across seeds -----
    logger.info("\n" + "=" * 80)
    logger.info(
        f"AGGREGATE: {config.dataset} ({config.n_encoder_seeds} seeds, "
        f"nbits={config.nbits}, max_depth={config.max_depth})"
    )
    logger.info("=" * 80)

    # Collect all representation names
    all_repr_names = list(per_seed_results[0]["representations"].keys())

    aggregate: dict[str, dict[str, float]] = {}
    for name in all_repr_names:
        test_accs = [s["representations"][name]["test_acc"] for s in per_seed_results]
        train_accs = [s["representations"][name]["train_acc"] for s in per_seed_results]
        bits = per_seed_results[0]["representations"][name]["bits_per_sample"]
        ratio = per_seed_results[0]["representations"][name]["compression_ratio"]

        aggregate[name] = {
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "train_acc_mean": float(np.mean(train_accs)),
            "train_acc_std": float(np.std(train_accs)),
            "bits_per_sample": float(bits),
            "compression_ratio": float(ratio),
        }

        # Aggregate purity metrics if present
        r0 = per_seed_results[0]["representations"][name]
        if "v_measure" in r0:
            v_measures = [
                s["representations"][name]["v_measure"] for s in per_seed_results
            ]
            aris = [s["representations"][name]["ari"] for s in per_seed_results]
            h_vals = [
                s["representations"][name]["h_y_given_z"] for s in per_seed_results
            ]
            aggregate[name]["v_measure_mean"] = float(np.mean(v_measures))
            aggregate[name]["v_measure_std"] = float(np.std(v_measures))
            aggregate[name]["ari_mean"] = float(np.mean(aris))
            aggregate[name]["ari_std"] = float(np.std(aris))
            aggregate[name]["h_y_given_z_mean"] = float(np.mean(h_vals))
            aggregate[name]["h_y_given_z_std"] = float(np.std(h_vals))

    # Log aggregate table
    raw_acc = aggregate["Raw-Embedding"]["test_acc_mean"]

    logger.info(f"\nCompression-Accuracy Tradeoff (raw baseline acc={raw_acc:.4f}):")
    header = (
        f"{'Representation':<25s}  {'Bits':>6s}  {'Ratio':>7s}  "
        f"{'TestAcc':>14s}  {'%ofRaw':>7s}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    sorted_names = sorted(
        all_repr_names,
        key=lambda k: aggregate[k]["bits_per_sample"],
    )
    for name in sorted_names:
        a = aggregate[name]
        pct_raw = a["test_acc_mean"] / raw_acc * 100 if raw_acc > 0 else 0
        logger.info(
            f"{name:<25s}  {a['bits_per_sample']:>6.0f}  "
            f"{a['compression_ratio']:>7.1f}x  "
            f"{a['test_acc_mean']:>.4f}±{a['test_acc_std']:.4f}  "
            f"{pct_raw:>6.1f}%"
        )

    # Log purity table
    logger.info(f"\nLabel Purity of Discrete Codes:")
    purity_header = f"{'Codes':<25s}  {'V-measure':>14s}  {'ARI':>14s}  {'H(Y|Z)':>14s}"
    logger.info(purity_header)
    logger.info("-" * len(purity_header))
    for name in sorted_names:
        a = aggregate[name]
        if "v_measure_mean" in a:
            logger.info(
                f"{name:<25s}  "
                f"{a['v_measure_mean']:>.4f}±{a['v_measure_std']:.4f}  "
                f"{a['ari_mean']:>.4f}±{a['ari_std']:.4f}  "
                f"{a['h_y_given_z_mean']:>.4f}±{a['h_y_given_z_std']:.4f}"
            )

    # Key comparison: RQ vs KMeans at matched bitrates
    logger.info(f"\nRQ vs KMeans Head-to-Head (matched bitrates):")
    for d_cut in range(1, config.max_depth + 1):
        rq_name = f"RQ-Codes-d{d_cut}"
        n_clusters = min(2 ** (d_cut * config.nbits), 10000)
        km_name = f"KM-Codes-k{n_clusters}"
        if rq_name in aggregate and km_name in aggregate:
            rq_acc = aggregate[rq_name]["test_acc_mean"]
            km_acc = aggregate[km_name]["test_acc_mean"]
            diff = rq_acc - km_acc
            winner = "RQ" if diff > 0 else "KM" if diff < 0 else "TIE"
            logger.info(
                f"  d_cut={d_cut}: RQ={rq_acc:.4f}, KM={km_acc:.4f}, "
                f"diff={diff:+.4f} ({winner})"
            )

        rq_r_name = f"RQ-Recon-d{d_cut}"
        km_r_name = f"KM-Recon-k{n_clusters}"
        if rq_r_name in aggregate and km_r_name in aggregate:
            rq_r_acc = aggregate[rq_r_name]["test_acc_mean"]
            km_r_acc = aggregate[km_r_name]["test_acc_mean"]
            diff = rq_r_acc - km_r_acc
            winner = "RQ" if diff > 0 else "KM" if diff < 0 else "TIE"
            logger.info(
                f"          Recon: RQ={rq_r_acc:.4f}, KM={km_r_acc:.4f}, "
                f"diff={diff:+.4f} ({winner})"
            )

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
        description="Compression-accuracy tradeoff: RQ trunk codes as "
        "sufficient statistics for classification"
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
        "--nbits",
        type=int,
        default=6,
        help="Bits per RQ level",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Max d_cut to evaluate",
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
        default="/tmp/compression_results",
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

    config = CompressionConfig(
        dataset=args.dataset,
        n_encoder_seeds=args.n_encoder_seeds,
        nbits=args.nbits,
        max_depth=args.max_depth,
        embedding_dim=args.embedding_dim,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
    )

    logger.info("=" * 60)
    logger.info("COMPRESSION-ACCURACY TRADEOFF EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset}")
    logger.info(
        f"n_encoder_seeds: {config.n_encoder_seeds}, nbits: {config.nbits}, "
        f"max_depth: {config.max_depth}"
    )

    start_time = time.time()
    results = run_compression_experiment(config)
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"compression_{config.dataset}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
