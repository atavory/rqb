#!/usr/bin/env python3


"""D_cut selection diagnostic on real tabular datasets.

Computes four independent criteria for selecting the trunk depth D_cut,
all of which can be evaluated *before* running any downstream bandit:

1. **Conditional entropy H(Y | trunk_codes)**: The label uncertainty remaining
   after observing trunk codes at each depth. Uses plug-in estimator with
   add-alpha smoothing on discrete codes — much more reliable than continuous
   MI estimators. The D_cut where H(Y|trunk) stops decreasing is the cut.

2. **Trunk reconstruction MSE**: MSE between original embeddings and trunk
   reconstructions at each depth. The elbow where marginal error reduction
   flattens marks where trunk information saturates.

3. **Context multiplicity**: Number of unique trunk codes and average samples
   per code at each depth. Given a budget T, choose D_cut so that
   T / n_unique_contexts >= min_obs for bandit convergence.

4. **Spectral gap**: Singular values of codebook centroids at each level.
   When the centroids at level l+1 contribute little spectral energy relative
   to level l, the marginal information gain is small.

Example:
    python3 scripts/experiments/run_dcut_diagnostic.py -- \\
        --dataset adult --max-depth 6
"""

from __future__ import annotations

import argparse
import hashlib
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
from modules.data import load_dataset
from modules.encoders.scarf import SCARFPretrainer
from modules.encoders.tab_transformer import MLPEncoder
from modules.embeddings import (
    EmbeddingConfig,
    EmbeddingExtractor,
)
from modules.features import (
    collect_raw_features,
    FeatureCache,
    normalize_for_rq,
    select_features_supervised,
    select_features_unsupervised,
)
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DcutDiagnosticConfig:
    """Configuration for D_cut diagnostic."""

    dataset: str = "adult"
    max_depth: int = 6  # Max RQ levels to evaluate
    nbits: int = 6  # Bits per RQ level
    nbits_values: list[int] | None = None  # Sweep over multiple nbits values
    alpha: float = 1.0  # Add-alpha smoothing for entropy

    # Encoder params
    embedding_dim: int = 64
    hidden_dim: int = 128
    encoder_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256

    # Bandit budget (for context multiplicity criterion)
    bandit_budget: int = 50000  # Expected number of bandit rounds
    min_obs_per_context: int = 10  # Minimum observations for convergence

    # Bootstrap/cross-validation
    n_bootstrap: int = 200  # Bootstrap samples for CIs
    n_cv_folds: int = 5  # Cross-validation folds
    bootstrap_ci: float = 0.95  # Confidence level for bootstrap CIs

    n_encoder_seeds: int = 3
    seed: int = 42
    device: str = "cpu"

    # Feature mode: "embeddings" (default), "raw", "selected", or "ssl"
    feature_mode: str = "embeddings"
    # Cache directory for raw features and feature selection results
    feature_cache_dir: str = "/tmp/feature_cache"


# =============================================================================
# Criterion 1: Conditional Entropy H(Y | trunk_codes)
# =============================================================================


def conditional_entropy(
    labels: np.ndarray,
    codes: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Compute H(Y | Z) using plug-in estimator with add-alpha smoothing.

    Args:
        labels: Discrete labels, shape (N,).
        codes: Trunk code tuples, shape (N, d_cut).
        alpha: Smoothing parameter (Laplace smoothing when alpha=1).

    Returns:
        Conditional entropy in nats.
    """
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)

    code_keys = [tuple(c) for c in codes]
    n = len(labels)
    unique_labels = set(labels.tolist())
    n_labels = len(unique_labels)

    # Count (code, label) pairs and code marginals
    joint_counts: dict[tuple[tuple[int, ...], int], int] = {}
    code_counts: dict[tuple[int, ...], int] = {}

    for code_key, label in zip(code_keys, labels.tolist()):
        joint_counts[(code_key, label)] = joint_counts.get((code_key, label), 0) + 1
        code_counts[code_key] = code_counts.get(code_key, 0) + 1

    # H(Y | Z) = sum_z P(z) * H(Y | Z=z)
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


def marginal_entropy(labels: np.ndarray, alpha: float = 1.0) -> float:
    """Compute H(Y) with add-alpha smoothing."""
    counts = Counter(labels.tolist())
    n = len(labels)
    n_classes = len(counts)
    total = n + alpha * n_classes

    h = 0.0
    for count in counts.values():
        p = (count + alpha) / total
        if p > 0:
            h -= p * math.log(p)
    return h


# =============================================================================
# Criterion 2: Trunk Reconstruction MSE
# =============================================================================


def compute_reconstruction_mse(
    embeddings_np: np.ndarray,
    rq: object,  # faiss.ResidualQuantizer
) -> float:
    """Compute MSE between embeddings and trunk reconstructions."""
    codes = rq.compute_codes(embeddings_np)  # pyre-ignore[16]
    recon = rq.decode(codes)  # pyre-ignore[16]
    return float(np.mean((embeddings_np - recon) ** 2))


# =============================================================================
# Criterion 3: Context Multiplicity
# =============================================================================


def compute_context_stats(
    codes: np.ndarray,
) -> tuple[int, float, float]:
    """Compute context multiplicity statistics.

    Returns:
        (n_unique, mean_samples_per_context, median_samples_per_context)
    """
    code_keys = [tuple(c) for c in codes]
    counts = Counter(code_keys)
    n_unique = len(counts)
    vals = list(counts.values())
    return n_unique, float(np.mean(vals)), float(np.median(vals))


# =============================================================================
# Criterion 4: Spectral Gap in Codebook Centroids
# =============================================================================


def compute_spectral_energy(
    embeddings_np: np.ndarray,
    rq: object,  # faiss.ResidualQuantizer
    d_cut: int,
) -> float:
    """Compute spectral energy (sum of squared singular values) of the
    trunk reconstruction centroids at depth d_cut.

    The marginal spectral energy of level l is:
        energy(1:l) - energy(1:l-1)
    """
    codes = rq.compute_codes(embeddings_np)  # pyre-ignore[16]
    recon = rq.decode(codes)  # pyre-ignore[16]
    # Center the reconstructions
    recon_centered = recon - recon.mean(axis=0)
    # SVD of centered reconstructions
    s = np.linalg.svd(recon_centered, compute_uv=False)
    return float(np.sum(s**2))


# =============================================================================
# Criterion 5: Bootstrap Confidence Intervals for H(Y|Z)
# =============================================================================


def bootstrap_conditional_entropy(
    labels: np.ndarray,
    codes: np.ndarray,
    alpha: float = 1.0,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for conditional entropy.

    Returns:
        (mean, ci_lower, ci_upper) for H(Y|Z).
    """
    rng = rng or np.random.RandomState(42)
    n = len(labels)
    boot_ents: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        h = conditional_entropy(labels[idx], codes[idx], alpha)
        boot_ents.append(h)

    boot_arr = np.array(boot_ents)
    alpha_half = (1 - ci) / 2
    lo = float(np.percentile(boot_arr, 100 * alpha_half))
    hi = float(np.percentile(boot_arr, 100 * (1 - alpha_half)))
    return float(boot_arr.mean()), lo, hi


# =============================================================================
# Criterion 6: Cross-Validated Conditional Entropy (detects overfitting)
# =============================================================================


def cv_conditional_entropy(
    labels: np.ndarray,
    codes: np.ndarray,
    alpha: float = 1.0,
    n_folds: int = 5,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float]:
    """Compute cross-validated conditional entropy to detect overfitting.

    For each fold:
    - Fit conditional probabilities P(Y|Z) on training data
    - Evaluate cross-entropy (negative log-likelihood) on held-out data

    Returns:
        (mean_cv_entropy, std_cv_entropy) across folds.
    """
    rng = rng or np.random.RandomState(42)
    n = len(labels)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    unique_labels = sorted(set(labels.tolist()))
    n_labels = len(unique_labels)

    cv_entropies: list[float] = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        # Fit P(Y|Z) on training data
        train_labels = labels[train_idx]
        train_codes = codes[train_idx]

        if train_codes.ndim == 1:
            train_codes = train_codes.reshape(-1, 1)

        code_keys_train = [tuple(c) for c in train_codes]
        joint_counts: dict[tuple[tuple[int, ...], int], int] = {}
        code_counts: dict[tuple[int, ...], int] = {}

        for ck, lab in zip(code_keys_train, train_labels.tolist()):
            joint_counts[(ck, lab)] = joint_counts.get((ck, lab), 0) + 1
            code_counts[ck] = code_counts.get(ck, 0) + 1

        # Evaluate on validation data
        val_labels = labels[val_idx]
        val_codes = codes[val_idx]
        if val_codes.ndim == 1:
            val_codes = val_codes.reshape(-1, 1)

        total_nll = 0.0
        n_val = len(val_labels)
        for i in range(n_val):
            ck = tuple(val_codes[i].tolist())
            lab = int(val_labels[i])
            cc = code_counts.get(ck, 0)
            jc = joint_counts.get((ck, lab), 0)
            # Smoothed probability
            p = (jc + alpha) / (cc + alpha * n_labels)
            total_nll -= math.log(max(p, 1e-10))

        cv_entropies.append(total_nll / n_val)

    return float(np.mean(cv_entropies)), float(np.std(cv_entropies))


# =============================================================================
# Criterion 7: Theoretical Bound (N/K^d feasibility)
# =============================================================================


def theoretical_bound_analysis(
    n_samples: int,
    n_classes: int,
    n_arms: int,
    nbits: int,
    max_depth: int,
    bandit_budget: int,
) -> dict[str, list[float]]:
    """Compute theoretical feasibility bounds for each d_cut.

    Key quantities:
    - K^d: Maximum number of contexts at depth d (K = 2^nbits)
    - N/K^d: Expected samples per context (given N training samples)
    - T/K^d: Expected bandit rounds per context (given budget T)
    - LinTS regret bound: O(d * sqrt(K^d * T * log(K^d * n_arms)))
      where d is the feature dimension for LinTS (d_cut-dependent)

    Returns:
        Dictionary with arrays indexed by depth 1..max_depth.
    """
    K = 2**nbits
    depths = list(range(1, max_depth + 1))

    max_contexts = [K**d for d in depths]
    samples_per_ctx_max = [n_samples / mc for mc in max_contexts]
    budget_per_ctx_max = [bandit_budget / mc for mc in max_contexts]

    # Bayesian regret lower bound: Omega(sqrt(K^d * n_arms * T))
    # This is the information-theoretic limit for K^d independent contexts
    regret_lower = [math.sqrt(mc * n_arms * bandit_budget) for mc in max_contexts]

    # Practical feasibility: need at least ~sqrt(T) observations per context
    # for LinTS to converge, so K^d <= T / sqrt(T) = sqrt(T)
    sqrt_T = math.sqrt(bandit_budget)
    feasible = [mc <= sqrt_T * 10 for mc in max_contexts]  # 10x margin

    return {
        "depths": depths,
        "max_contexts": max_contexts,
        "samples_per_ctx_max": samples_per_ctx_max,
        "budget_per_ctx_max": budget_per_ctx_max,
        "regret_lower_bound": regret_lower,
        "feasible": feasible,
    }


# =============================================================================
# Criterion 8: Fano's Inequality (classification error lower bound)
# =============================================================================


def fano_error_bound(
    h_y_given_z: float,
    n_classes: int,
) -> float:
    """Compute Fano's inequality lower bound on classification error.

    For |Y| >= 3: P_e >= (H(Y|Z) - log(2)) / log(|Y| - 1)
    For |Y| = 2: Use the binary inverse relation P_e >= h_b^{-1}(H(Y|Z))
    where h_b(p) = -p*log(p) - (1-p)*log(1-p) is binary entropy.

    Since h_b^{-1} has no closed form, we approximate by bisection.

    Returns:
        Lower bound on Bayes error rate (in [0, 1]).
    """
    if n_classes <= 1:
        return 0.0

    if n_classes == 2:
        # Binary case: find p such that h_b(p) = H(Y|Z), with p <= 0.5
        # h_b(p) = -p*log(p) - (1-p)*log(1-p), max at p=0.5 = log(2) ≈ 0.693
        target = min(h_y_given_z, math.log(2))
        if target <= 0:
            return 0.0
        # Bisection on [0, 0.5]
        lo, hi = 0.0, 0.5
        for _ in range(50):
            mid = (lo + hi) / 2
            if mid <= 0 or mid >= 1:
                break
            h_b = -mid * math.log(mid) - (1 - mid) * math.log(1 - mid)
            if h_b < target:
                lo = mid
            else:
                hi = mid
        return max(0.0, min(0.5, (lo + hi) / 2))

    # Multiclass: standard Fano
    log_m_minus_1 = math.log(n_classes - 1)
    bound = (h_y_given_z - math.log(2)) / log_m_minus_1
    return max(0.0, min(1.0, bound))


# =============================================================================
# Criterion 9: Normalized Mutual Information (NMI) for cross-dataset comparison
# =============================================================================


def normalized_mutual_info(
    h_y: float,
    h_y_given_z: float,
    labels: np.ndarray,
    codes: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Compute NMI(Y; Z) = I(Y;Z) / sqrt(H(Y) * H(Z)).

    NMI is a scale-free measure in [0,1] that enables comparison across
    datasets with different numbers of classes and different code vocabularies.

    Returns:
        NMI score in [0, 1].
    """
    i_y_z = max(h_y - h_y_given_z, 0.0)

    # Compute H(Z) — entropy of the code distribution
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)
    code_keys = [tuple(c) for c in codes]
    counts = Counter(code_keys)
    n = len(code_keys)
    h_z = 0.0
    for count in counts.values():
        p = (count + alpha) / (n + alpha * len(counts))
        if p > 0:
            h_z -= p * math.log(p)

    denom = math.sqrt(h_y * h_z) if h_y > 0 and h_z > 0 else 1.0
    return i_y_z / denom


# =============================================================================
# Criterion 10: Information Efficiency (bits per RQ level)
# =============================================================================


def information_efficiency(
    h_y: float,
    h_y_given_z: float,
    d_cut: int,
    nbits: int,
) -> float:
    """Compute information efficiency: label-info captured per code bit.

    efficiency = I(Y; Z) / (d_cut * nbits)

    This measures how efficiently the RQ code captures label-relevant
    information. Higher is better. Diminishing efficiency signals that
    adding more trunk levels wastes bits on noise.

    Returns:
        Information efficiency in nats per bit.
    """
    i_y_z = max(h_y - h_y_given_z, 0.0)
    total_bits = d_cut * nbits
    return i_y_z / total_bits if total_bits > 0 else 0.0


# =============================================================================
# Criterion 11: Effective Dimensionality of Trunk Reconstructions
# =============================================================================


def effective_dimensionality(
    embeddings_np: np.ndarray,
    rq: object,  # faiss.ResidualQuantizer
) -> float:
    """Compute effective dimensionality of trunk reconstructions.

    Uses the participation ratio:
        d_eff = (sum lambda_i)^2 / sum(lambda_i^2)

    where lambda_i are eigenvalues of the covariance matrix. d_eff = 1
    when all variance is in one direction (maximally compressed), and
    d_eff = d when variance is uniform (no compression).

    This quantifies how much the trunk compresses the embedding space.
    Lower d_eff at deeper trunk means more aggressive compression.

    Returns:
        Effective dimensionality (participation ratio).
    """
    codes = rq.compute_codes(embeddings_np)  # pyre-ignore[16]
    recon = rq.decode(codes)  # pyre-ignore[16]
    recon_centered = recon - recon.mean(axis=0)

    # Eigenvalues of covariance matrix via SVD
    s = np.linalg.svd(recon_centered, compute_uv=False)
    eigenvalues = s**2 / (len(recon_centered) - 1)

    # Participation ratio
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues**2).sum()
    if sum_eig_sq == 0:
        return 0.0
    return float(sum_eig**2 / sum_eig_sq)


# =============================================================================
# Main Diagnostic
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


def _encoder_cache_key(
    config: DcutDiagnosticConfig,
    encoder_seed: int,
) -> str:
    """Deterministic hash for encoder training config."""
    key_dict = {
        "dataset": config.dataset,
        "encoder_seed": encoder_seed,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "encoder_epochs": config.encoder_epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


_DCUT_CACHE_DIR = Path("/tmp/dcut_cache")


def _train_encoder_and_collect_embeddings(
    config: DcutDiagnosticConfig,
    train_dataset: object,
    metadata: object,
    n_classes: int,
    encoder_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an MLP encoder and return (embeddings_np, labels_np).

    Caches encoder weights and embeddings to /tmp/dcut_cache/ keyed by
    (dataset, seed, encoder hyperparameters). On cache hit, skips training
    entirely and loads precomputed embeddings.
    """
    cache_hash = _encoder_cache_key(config, encoder_seed)
    cache_dir = _DCUT_CACHE_DIR / config.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / f"embeddings_{cache_hash}.npz"
    weights_path = cache_dir / f"encoder_{cache_hash}.pt"

    # Check for cached embeddings
    if embeddings_path.exists():
        logger.info(f"  Cache hit: loading embeddings from {embeddings_path}")
        data = np.load(embeddings_path)
        return data["embeddings"], data["labels"]

    # No cache — train from scratch
    np.random.seed(encoder_seed)
    torch.manual_seed(encoder_seed)

    train_loader: DataLoader = DataLoader(  # pyre-ignore[11]
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(encoder_seed),
    )

    encoder = MLPEncoder(
        n_categories=metadata.category_sizes,
        n_continuous=metadata.n_continuous,
        dim=config.embedding_dim,
        hidden_dims=[config.hidden_dim, config.hidden_dim],
    ).to(device)

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
            loss = criterion(classifier(embeddings), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}"
            )

    all_emb: list[Tensor] = []
    all_labels: list[Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_emb.append(emb.cpu())
            all_labels.append(batch_y)

    embeddings_t = torch.cat(all_emb)
    labels_t = torch.cat(all_labels)
    embeddings_np = embeddings_t.detach().numpy().astype(np.float32)
    labels_np = labels_t.numpy()

    # Save cache
    try:
        np.savez(embeddings_path, embeddings=embeddings_np, labels=labels_np)
        torch.save(encoder.state_dict(), weights_path)
        logger.info(f"  Cached embeddings to {embeddings_path}")
    except Exception as e:
        logger.warning(f"  Failed to write cache: {e}")

    return embeddings_np, labels_np


def _compute_depth_diagnostics(
    config: DcutDiagnosticConfig,
    embeddings_np: np.ndarray,
    labels_np: np.ndarray,
    h_y: float,
    nbits_val: int,
    depths: list[int],
    encoder_seed: int,
) -> dict[str, list[float]]:
    """Compute all diagnostic metrics across depths for one encoder seed."""
    import faiss

    seed_metrics: dict[str, list[float]] = {
        "cond_entropy": [],
        "recon_mse": [],
        "n_unique_contexts": [],
        "mean_samples_per_ctx": [],
        "spectral_energy": [],
        "bootstrap_ci_lo": [],
        "bootstrap_ci_hi": [],
        "cv_entropy_mean": [],
        "cv_entropy_std": [],
        "fano_error_bound": [],
        "nmi": [],
        "info_efficiency": [],
        "effective_dim": [],
    }

    n_classes = len(set(labels_np.tolist()))

    for d_cut in depths:
        print(
            f"  [d_cut={d_cut}] Training RQ: dim={embeddings_np.shape[1]}, "
            f"nbits={nbits_val}, N={embeddings_np.shape[0]}",
            flush=True,
        )
        logger.info(f"  d_cut={d_cut}:")

        rq = faiss.ResidualQuantizer(embeddings_np.shape[1], d_cut, nbits_val)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.verbose = False
        rq.train(embeddings_np)
        print(f"  [d_cut={d_cut}] RQ trained, computing metrics...", flush=True)

        codes_raw = rq.compute_codes(embeddings_np)
        codes_int = codes_raw.astype(np.int64)

        h_y_given_z = conditional_entropy(labels_np, codes_int, config.alpha)
        info_captured = h_y - h_y_given_z
        seed_metrics["cond_entropy"].append(h_y_given_z)
        logger.info(
            f"    H(Y|Z) = {h_y_given_z:.4f}, info captured = {info_captured:.4f} nats"
        )

        mse = compute_reconstruction_mse(embeddings_np, rq)
        seed_metrics["recon_mse"].append(mse)
        logger.info(f"    Recon MSE = {mse:.6f}")

        n_unique, mean_spc, _median_spc = compute_context_stats(codes_int)
        seed_metrics["n_unique_contexts"].append(float(n_unique))
        seed_metrics["mean_samples_per_ctx"].append(mean_spc)
        budget_ratio = config.bandit_budget / max(n_unique, 1)
        feasible = budget_ratio >= config.min_obs_per_context
        logger.info(
            f"    Contexts: {n_unique}, mean samples/ctx = {mean_spc:.1f}, "
            f"budget ratio = {budget_ratio:.1f} {'OK' if feasible else 'TOO LOW'}"
        )

        energy = compute_spectral_energy(embeddings_np, rq, d_cut)
        seed_metrics["spectral_energy"].append(energy)
        logger.info(f"    Spectral energy = {energy:.2f}")

        boot_rng = np.random.RandomState(encoder_seed + d_cut)
        _, ci_lo, ci_hi = bootstrap_conditional_entropy(
            labels_np,
            codes_int,
            config.alpha,
            config.n_bootstrap,
            config.bootstrap_ci,
            boot_rng,
        )
        seed_metrics["bootstrap_ci_lo"].append(ci_lo)
        seed_metrics["bootstrap_ci_hi"].append(ci_hi)
        logger.info(
            f"    Bootstrap {config.bootstrap_ci:.0%} CI: [{ci_lo:.4f}, {ci_hi:.4f}]"
        )

        cv_rng = np.random.RandomState(encoder_seed + d_cut + 100)
        cv_mean, cv_std = cv_conditional_entropy(
            labels_np, codes_int, config.alpha, config.n_cv_folds, cv_rng
        )
        seed_metrics["cv_entropy_mean"].append(cv_mean)
        seed_metrics["cv_entropy_std"].append(cv_std)
        overfit_gap = cv_mean - h_y_given_z
        logger.info(
            f"    CV entropy: {cv_mean:.4f} +/- {cv_std:.4f} "
            f"(overfitting gap: {overfit_gap:.4f})"
        )

        fano_bound = fano_error_bound(h_y_given_z, n_classes)
        seed_metrics["fano_error_bound"].append(fano_bound)
        logger.info(f"    Fano error bound: {fano_bound:.4f}")

        nmi_val = normalized_mutual_info(
            h_y, h_y_given_z, labels_np, codes_int, config.alpha
        )
        seed_metrics["nmi"].append(nmi_val)
        logger.info(f"    NMI(Y; Z): {nmi_val:.4f}")

        info_eff = information_efficiency(h_y, h_y_given_z, d_cut, nbits_val)
        seed_metrics["info_efficiency"].append(info_eff)
        logger.info(
            f"    Info efficiency: {info_eff:.4f} nats/bit "
            f"({info_eff * d_cut * nbits_val / max(h_y, 1e-10) * 100:.1f}% of H(Y))"
        )

        eff_dim = effective_dimensionality(embeddings_np, rq)
        ambient_dim = embeddings_np.shape[1]
        seed_metrics["effective_dim"].append(eff_dim)
        logger.info(
            f"    Effective dim: {eff_dim:.1f} / {ambient_dim} "
            f"({eff_dim / ambient_dim * 100:.0f}% of ambient)"
        )

    return seed_metrics


def _log_nbits_summary(
    config: DcutDiagnosticConfig,
    means: dict[str, np.ndarray],
    depths: list[int],
    h_y_mean: float,
    labels_np: np.ndarray,
    nbits_val: int,
    recommended_cut: int,
    marginal_info: list[float],
    theory: dict[str, object],
    nbits_tag: str,
    actual_dim: int,
) -> None:
    """Log diagnostic summary tables for one nbits value."""
    logger.info("\n" + "=" * 80)
    logger.info(f"D_CUT DIAGNOSTIC SUMMARY: {config.dataset}{nbits_tag}")
    logger.info("=" * 80)

    header = (
        f"{'d_cut':>5s}  {'H(Y|Z)':>8s}  {'info%':>6s}  "
        f"{'MSE':>10s}  {'#ctx':>6s}  {'samp/ctx':>8s}  "
        f"{'T/ctx':>6s}  {'spectral':>10s}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for i, d_cut in enumerate(depths):
        info_pct = 100 * (h_y_mean - means["cond_entropy"][i]) / max(h_y_mean, 1e-10)
        budget_ratio = config.bandit_budget / max(means["n_unique_contexts"][i], 1)
        logger.info(
            f"{d_cut:5d}  {means['cond_entropy'][i]:8.4f}  {info_pct:5.1f}%  "
            f"{means['recon_mse'][i]:10.6f}  {means['n_unique_contexts'][i]:6.0f}  "
            f"{means['mean_samples_per_ctx'][i]:8.1f}  {budget_ratio:6.1f}  "
            f"{means['spectral_energy'][i]:10.2f}"
        )

    logger.info(f"\nRECOMMENDED D_CUT: {recommended_cut}")
    logger.info(
        f"  Based on: conditional entropy elbow + feasibility for T={config.bandit_budget}"
    )

    logger.info("\nMarginal information gain per level:")
    for i, d_cut in enumerate(depths):
        mi = marginal_info[i]
        pct = 100 * mi / max(h_y_mean, 1e-10)
        bar = "#" * int(max(pct, 0))
        logger.info(f"  d_cut={d_cut}: {mi:.4f} nats ({pct:5.1f}%) {bar}")

    logger.info("\nCross-validation analysis (overfitting detection):")
    logger.info(
        f"{'d_cut':>5s}  {'H(Y|Z)':>8s}  {'CV H(Y|Z)':>10s}  "
        f"{'gap':>8s}  {'boot CI':>20s}  {'overfit?':>8s}"
    )
    for i, d_cut in enumerate(depths):
        h_plug = means["cond_entropy"][i]
        h_cv = means["cv_entropy_mean"][i]
        gap = h_cv - h_plug
        boot_lo = means["bootstrap_ci_lo"][i]
        boot_hi = means["bootstrap_ci_hi"][i]
        overfit = "YES" if gap > 0.05 * h_y_mean else "no"
        logger.info(
            f"{d_cut:5d}  {h_plug:8.4f}  {h_cv:10.4f}  "
            f"{gap:8.4f}  [{boot_lo:.4f}, {boot_hi:.4f}]  {overfit:>8s}"
        )

    n_samples = len(labels_np)
    n_classes = len(set(labels_np.tolist()))
    n_arms = max(5, n_classes)
    logger.info("\nTheoretical feasibility bounds:")
    logger.info(
        f"{'d_cut':>5s}  {'K^d (max ctx)':>12s}  {'N/K^d':>8s}  "
        f"{'T/K^d':>8s}  {'regret LB':>10s}  {'feasible':>8s}"
    )
    for i, d_cut in enumerate(depths):
        logger.info(
            f"{d_cut:5d}  {theory['max_contexts'][i]:12d}  "
            f"{theory['samples_per_ctx_max'][i]:8.1f}  "
            f"{theory['budget_per_ctx_max'][i]:8.1f}  "
            f"{theory['regret_lower_bound'][i]:10.1f}  "
            f"{'YES' if theory['feasible'][i] else 'NO':>8s}"
        )

    logger.info("\nBias-Variance Tradeoff (d_cut -> regret):")
    logger.info(
        "  The d_cut parameter controls a bias-variance tradeoff in contextual bandits:"
    )
    logger.info(
        "  - BIAS: Deeper trunk (higher d_cut) -> finer context granularity -> lower"
    )
    logger.info(
        "    approximation error, but reconstruction is lossier (more MSE in tail)"
    )
    logger.info(
        "  - VARIANCE: More contexts -> fewer observations per context -> higher"
    )
    logger.info("    posterior uncertainty -> slower convergence -> higher regret")
    logger.info(
        f"  At d_cut=1: {means['n_unique_contexts'][0]:.0f} contexts, "
        f"{means['mean_samples_per_ctx'][0]:.1f} samples/ctx -> low variance, low bias"
    )
    if len(depths) >= 3:
        logger.info(
            f"  At d_cut=3: {means['n_unique_contexts'][2]:.0f} contexts, "
            f"{means['mean_samples_per_ctx'][2]:.1f} samples/ctx -> high variance"
        )

    logger.info("\nFano's Inequality (Bayes error lower bound):")
    logger.info("  Fano's inequality: P_e >= (H(Y|Z) - log 2) / log(|Y|-1)")
    logger.info("  This gives a fundamental lower bound on classification error")
    logger.info(
        "  given access to trunk codes. Lower bound -> trunk is more informative."
    )
    logger.info(f"{'d_cut':>5s}  {'Fano LB':>8s}  {'1-Fano':>8s}")
    for i, d_cut in enumerate(depths):
        fano = means["fano_error_bound"][i]
        logger.info(f"{d_cut:5d}  {fano:8.4f}  {1 - fano:8.4f}")

    logger.info(
        "\nInformation-theoretic summary (NMI = scale-free, Efficiency = nats/bit):"
    )
    logger.info(
        f"{'d_cut':>5s}  {'NMI':>8s}  {'eff':>10s}  {'eff_dim':>8s}  {'dim_ratio':>9s}"
    )
    for i, d_cut in enumerate(depths):
        logger.info(
            f"{d_cut:5d}  {means['nmi'][i]:8.4f}  "
            f"{means['info_efficiency'][i]:10.4f}  "
            f"{means['effective_dim'][i]:8.1f}  "
            f"{means['effective_dim'][i] / actual_dim:9.2%}"
        )


def run_dcut_diagnostic(config: DcutDiagnosticConfig) -> dict[str, object]:
    """Run full D_cut diagnostic on a real dataset.

    When config.nbits_values is set, runs the full diagnostic for each nbits
    value and returns per-nbits results under a "nbits_sweep" key.
    """
    device = torch.device(config.device)

    # Load data
    print(f"Loading dataset: {config.dataset}", flush=True)
    logger.info(f"Loading dataset: {config.dataset}")
    train_dataset, _val, _test, metadata = load_dataset(config.dataset)
    n_classes = metadata.n_classes
    logger.info(
        f"Loaded: {metadata.n_samples} samples, {n_classes} classes, "
        f"{metadata.n_continuous} cont, {metadata.n_categorical} cat"
    )

    nbits_list = config.nbits_values or [config.nbits]
    depths = list(range(1, config.max_depth + 1))

    # Per-nbits results (only populated when sweeping)
    nbits_sweep_results: dict[int, dict[str, object]] = {}

    for nbits_val in nbits_list:
        if len(nbits_list) > 1:
            logger.info(f"\n{'#' * 60}")
            logger.info(f"# nbits = {nbits_val}")
            logger.info(f"{'#' * 60}")

        # Results across encoder seeds
        all_results: dict[str, list[list[float]]] = {
            k: []
            for k in [
                "cond_entropy",
                "recon_mse",
                "n_unique_contexts",
                "mean_samples_per_ctx",
                "spectral_energy",
                "bootstrap_ci_lo",
                "bootstrap_ci_hi",
                "cv_entropy_mean",
                "cv_entropy_std",
                "fano_error_bound",
                "nmi",
                "info_efficiency",
                "effective_dim",
            ]
        }

        labels_np = None
        actual_dim = config.embedding_dim  # Updated below for raw/selected

        if config.feature_mode in ("raw", "selected"):
            # Raw/selected features: no encoder, single iteration
            cache = FeatureCache(config.feature_cache_dir)
            features_np, labels_np = cache.get_or_compute_raw_features(
                config.dataset,
                lambda: collect_raw_features(train_dataset),
            )

            if config.feature_mode == "selected":
                selected_indices, mi_scores = cache.get_or_compute_selection(
                    config.dataset,
                    "supervised",
                    lambda: select_features_supervised(features_np, labels_np),
                )
                features_np = features_np[:, selected_indices]
                logger.info(
                    f"  Feature selection: {len(selected_indices)} of "
                    f"{mi_scores.shape[0]} features selected (supervised MI)"
                )

            # Normalize before RQ
            features_np, _, _ = normalize_for_rq(features_np)
            actual_dim = features_np.shape[1]
            print(
                f"  Features ready: {features_np.shape[0]} x {actual_dim}, "
                f"mode={config.feature_mode}, H(Y)={marginal_entropy(labels_np, config.alpha):.4f}",
                flush=True,
            )

            h_y = marginal_entropy(labels_np, config.alpha)
            logger.info(
                f"  Raw features: {features_np.shape[0]} samples, "
                f"{actual_dim} dims, H(Y) = {h_y:.4f} nats"
            )

            seed_metrics = _compute_depth_diagnostics(
                config,
                features_np,
                labels_np,
                h_y,
                nbits_val,
                depths,
                config.seed,
            )

            for k, v in seed_metrics.items():
                all_results[k].append(v)

        elif config.feature_mode == "ssl":
            # SSL mode: self-supervised SCARF embeddings (no labels used)
            cache = FeatureCache(config.feature_cache_dir)
            features_np, labels_np = cache.get_or_compute_raw_features(
                config.dataset,
                lambda: collect_raw_features(train_dataset),
            )

            input_dim = features_np.shape[1]
            logger.info(
                f"  SSL mode: {features_np.shape[0]} samples, "
                f"{input_dim} raw features -> SCARF pre-training"
            )

            # Cache key for SCARF encoder
            scarf_cache_key = _encoder_cache_key(config, config.seed)
            scarf_cache_dir = _DCUT_CACHE_DIR / config.dataset
            scarf_cache_dir.mkdir(parents=True, exist_ok=True)
            scarf_emb_path = scarf_cache_dir / f"scarf_embeddings_{scarf_cache_key}.npz"
            scarf_ckpt_path = scarf_cache_dir / f"scarf_encoder_{scarf_cache_key}.pt"

            if scarf_emb_path.exists() and scarf_ckpt_path.exists():
                logger.info(f"  SCARF cache hit: {scarf_emb_path}")
                logger.info(f"  SCARF encoder checkpoint: {scarf_ckpt_path}")
                data = np.load(scarf_emb_path)
                ssl_embeddings = data["embeddings"]
            else:
                # Normalize raw features before SCARF training
                features_scaled, _, _ = normalize_for_rq(features_np)

                pretrainer = SCARFPretrainer(
                    input_dim=input_dim,
                    embedding_dim=config.embedding_dim,
                    hidden_dims=[config.hidden_dim, config.hidden_dim],
                    corruption_rate=0.3,
                    temperature=0.1,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    batch_size=config.batch_size,
                )
                print(
                    f"  Pre-training SCARF encoder: "
                    f"dim={config.embedding_dim}, epochs={config.encoder_epochs}",
                    flush=True,
                )
                pretrainer.fit(
                    features_scaled,
                    epochs=config.encoder_epochs,
                    device=config.device,
                    seed=config.seed,
                )
                ssl_embeddings = pretrainer.transform(
                    features_scaled, device=config.device
                )
                # Cache embeddings and encoder checkpoint
                try:
                    np.savez(scarf_emb_path, embeddings=ssl_embeddings)
                    pretrainer.save_checkpoint(scarf_ckpt_path)
                    logger.info(f"  Cached SCARF embeddings to {scarf_emb_path}")
                    logger.info(f"  Saved SCARF encoder to {scarf_ckpt_path}")
                except Exception as e:
                    logger.warning(f"  Failed to cache SCARF artifacts: {e}")

            # Normalize embeddings before RQ
            ssl_embeddings, _, _ = normalize_for_rq(ssl_embeddings)
            actual_dim = ssl_embeddings.shape[1]

            h_y = marginal_entropy(labels_np, config.alpha)
            print(
                f"  SSL embeddings ready: {ssl_embeddings.shape[0]} x {actual_dim}, "
                f"H(Y)={h_y:.4f}",
                flush=True,
            )
            logger.info(
                f"  SSL embeddings: {ssl_embeddings.shape[0]} samples, "
                f"{actual_dim} dims, H(Y) = {h_y:.4f} nats"
            )

            seed_metrics = _compute_depth_diagnostics(
                config,
                ssl_embeddings,
                labels_np,
                h_y,
                nbits_val,
                depths,
                config.seed,
            )

            for k, v in seed_metrics.items():
                all_results[k].append(v)

        elif config.feature_mode in ("iforest", "lgbm"):
            # Tree-based embeddings: iforest (unsupervised) or lgbm (supervised)
            cache = FeatureCache(config.feature_cache_dir)
            features_np, labels_np = cache.get_or_compute_raw_features(
                config.dataset,
                lambda: collect_raw_features(train_dataset),
            )

            emb_config = EmbeddingConfig(embedding_type=config.feature_mode)
            extractor = EmbeddingExtractor(emb_config)
            tree_embeddings = extractor.fit_transform(
                features_np,
                y_train=labels_np if config.feature_mode == "lgbm" else None,
            )
            tree_embeddings, _, _ = normalize_for_rq(tree_embeddings)
            actual_dim = tree_embeddings.shape[1]

            h_y = marginal_entropy(labels_np, config.alpha)
            logger.info(
                f"  {config.feature_mode} embeddings: "
                f"{tree_embeddings.shape[0]} samples, {actual_dim} dims, "
                f"H(Y) = {h_y:.4f} nats"
            )

            seed_metrics = _compute_depth_diagnostics(
                config,
                tree_embeddings,
                labels_np,
                h_y,
                nbits_val,
                depths,
                config.seed,
            )

            for k, v in seed_metrics.items():
                all_results[k].append(v)

        else:
            # Embeddings mode: train encoder, iterate over seeds
            for enc_idx in range(config.n_encoder_seeds):
                encoder_seed = config.seed + enc_idx * 10000
                logger.info(
                    f"\nEncoder seed {enc_idx + 1}/{config.n_encoder_seeds} (seed={encoder_seed})"
                )

                embeddings_np, labels_np = _train_encoder_and_collect_embeddings(
                    config,
                    train_dataset,
                    metadata,
                    n_classes,
                    encoder_seed,
                    device,
                )

                # Standardize embeddings before RQ (critical for codebook quality)
                embeddings_np, _, _ = normalize_for_rq(embeddings_np)
                actual_dim = embeddings_np.shape[1]

                h_y = marginal_entropy(labels_np, config.alpha)
                logger.info(f"  H(Y) = {h_y:.4f} nats")

                seed_metrics = _compute_depth_diagnostics(
                    config,
                    embeddings_np,
                    labels_np,
                    h_y,
                    nbits_val,
                    depths,
                    encoder_seed,
                )

                for k, v in seed_metrics.items():
                    all_results[k].append(v)

        # Aggregate across seeds
        nbits_tag = f" (nbits={nbits_val})" if len(nbits_list) > 1 else ""

        arrays = {k: np.array(v) for k, v in all_results.items()}
        means = {k: v.mean(axis=0) for k, v in arrays.items()}
        stds = {k: v.std(axis=0) for k, v in arrays.items()}

        h_y_mean = marginal_entropy(labels_np, config.alpha)

        # Recommendation based on conditional entropy elbow
        ce_means = means["cond_entropy"]
        marginal_info = [
            ce_means[i - 1] - ce_means[i] if i > 0 else h_y_mean - ce_means[0]
            for i in range(len(depths))
        ]
        threshold = 0.1 * marginal_info[0] if marginal_info[0] > 0 else 0.01
        recommended_cut = depths[-1]
        for i, mi in enumerate(marginal_info):
            if i > 0 and mi < threshold:
                recommended_cut = depths[i - 1]
                break

        # Check feasibility
        for i, d_cut in enumerate(depths):
            budget_ratio = config.bandit_budget / max(means["n_unique_contexts"][i], 1)
            if budget_ratio < config.min_obs_per_context:
                feasible_max = depths[i - 1] if i > 0 else depths[0]
                if feasible_max < recommended_cut:
                    logger.info(
                        f"\nFeasibility constraint: d_cut={d_cut} has too many contexts "
                        f"for budget T={config.bandit_budget}. Max feasible: {feasible_max}"
                    )
                    recommended_cut = min(recommended_cut, feasible_max)
                break

        # Theoretical bounds
        n_samples = len(labels_np)
        n_classes_actual = len(set(labels_np.tolist()))
        n_arms = max(5, n_classes_actual)
        theory = theoretical_bound_analysis(
            n_samples,
            n_classes_actual,
            n_arms,
            nbits_val,
            config.max_depth,
            config.bandit_budget,
        )

        _log_nbits_summary(
            config,
            means,
            depths,
            h_y_mean,
            labels_np,
            nbits_val,
            recommended_cut,
            marginal_info,
            theory,
            nbits_tag,
            actual_dim,
        )

        # Build per-nbits result
        nbits_result: dict[str, object] = {
            "nbits": nbits_val,
            "depths": depths,
            "h_y": h_y_mean,
            "recommended_dcut": recommended_cut,
            "theoretical_bounds": theory,
            "feature_mode": config.feature_mode,
            "actual_dim": actual_dim,
        }
        for k in all_results:
            nbits_result[f"{k}_mean"] = means[k].tolist()
            nbits_result[f"{k}_std"] = stds[k].tolist()
        nbits_result["marginal_info"] = marginal_info

        nbits_sweep_results[nbits_val] = nbits_result

    # If only one nbits value, return flat result (backward-compatible)
    if len(nbits_list) == 1:
        single = nbits_sweep_results[nbits_list[0]]
        result: dict[str, object] = {
            "dataset": config.dataset,
            "config": asdict(config),
        }
        result.update(single)
        return result

    # Multi-nbits sweep: log summary and return all
    logger.info("\n" + "=" * 80)
    logger.info(f"NBITS SWEEP SUMMARY: {config.dataset}")
    logger.info("=" * 80)
    logger.info(
        f"{'nbits':>5s}  {'rec d_cut':>9s}  {'H(Y|Z)@d1':>10s}  {'#ctx@d1':>8s}"
    )
    for nb in nbits_list:
        nr = nbits_sweep_results[nb]
        logger.info(
            f"{nb:5d}  {nr['recommended_dcut']:>9d}  "
            f"{nr['cond_entropy_mean'][0]:10.4f}  "
            f"{nr['n_unique_contexts_mean'][0]:8.0f}"
        )

    result = {
        "dataset": config.dataset,
        "nbits_values": nbits_list,
        "config": asdict(config),
        "nbits_sweep": {str(nb): nbits_sweep_results[nb] for nb in nbits_list},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D_cut selection diagnostic: conditional entropy + reconstruction + multiplicity + spectral"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=[
            "adult",
            "bank_marketing",
            "german_credit",
            "covertype",
            "higgs",
            "helena",
            "jannis",
            "year_prediction",
            "click",
            "aloi",
            "letter",
            "dionis",
            "volkert",
        ],
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="Max RQ depth to evaluate"
    )
    parser.add_argument("--nbits", type=int, default=6, help="Bits per RQ level")
    parser.add_argument(
        "--nbits-values",
        type=int,
        nargs="+",
        default=None,
        help="Sweep over multiple nbits values (overrides --nbits when set)",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--encoder-epochs", type=int, default=20)
    parser.add_argument("--n-encoder-seeds", type=int, default=3)
    parser.add_argument(
        "--bandit-budget", type=int, default=50000, help="Expected bandit rounds T"
    )
    parser.add_argument(
        "--min-obs", type=int, default=10, help="Min observations per context"
    )
    parser.add_argument("--output-dir", type=str, default="/tmp/dcut_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for encoder training (cpu or cuda)",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="embeddings",
        choices=["embeddings", "raw", "selected", "ssl", "iforest", "lgbm"],
        help="Input space for RQ: encoder embeddings, raw features, selected features, SSL (SCARF) embeddings, or tree-based embeddings (iforest/lgbm)",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="/tmp/feature_cache",
        help="Directory for caching raw features and feature selection results",
    )
    args = parser.parse_args()

    config = DcutDiagnosticConfig(
        dataset=args.dataset,
        max_depth=args.max_depth,
        nbits=args.nbits,
        nbits_values=args.nbits_values,
        embedding_dim=args.embedding_dim,
        encoder_epochs=args.encoder_epochs,
        n_encoder_seeds=args.n_encoder_seeds,
        bandit_budget=args.bandit_budget,
        min_obs_per_context=args.min_obs,
        seed=args.seed,
        device=args.device,
        feature_mode=args.feature_mode,
        feature_cache_dir=args.feature_cache_dir,
    )

    logger.info("=" * 60)
    logger.info("D_CUT SELECTION DIAGNOSTIC")
    logger.info("=" * 60)
    logger.info(
        f"Dataset: {config.dataset}, max_depth: {config.max_depth}, "
        f"feature_mode: {config.feature_mode}"
    )
    logger.info(
        f"Bandit budget T={config.bandit_budget}, min_obs={config.min_obs_per_context}"
    )

    start = time.time()
    results = run_dcut_diagnostic(config)
    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{config.feature_mode}" if config.feature_mode != "embeddings" else ""
    nbits_tag = f"_nbits{config.nbits}" if config.nbits_values is None else ""
    output_file = (
        output_dir / f"dcut_diagnostic_{config.dataset}{suffix}{nbits_tag}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Results saved to: {output_file}", flush=True)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
