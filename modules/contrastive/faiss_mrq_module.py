#!/usr/bin/env python3


"""Faiss-based m-RQ (Residual Quantization) module for contrastive learning.

Uses faiss.ResidualQuantizer with GPU acceleration for fast codebook fitting.
Implements the trunk-tail architecture from the paper:
- Shared trunk RQ fitted on ALL data (class-agnostic structure)
- Per-class tail RQs fitted on class-specific residuals
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np
import torch
import logging
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class FaissMRQConfig:
    """Configuration for Faiss-based m-RQ."""

    n_levels: int = 8
    nbits: int = 6  # bits per level (K = 2^nbits centroids)
    d_cut: int = 4  # trunk levels (rest are tail)
    use_gpu: bool = True
    beam_size: int = 5
    niter: int = 25  # k-means iterations


class FaissMRQModule:
    """m-RQ using faiss.ResidualQuantizer with GPU acceleration.

    Architecture:
        - Shared trunk RQ: d_cut levels fitted on ALL data
        - Per-class tail RQs: (n_levels - d_cut) levels fitted per class

    The trunk captures class-agnostic structure, while tails capture
    class-specific variations.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        config: FaissMRQConfig,
    ) -> None:
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.config = config
        self.n_tail_levels = config.n_levels - config.d_cut

        # Shared trunk RQ (fitted on ALL data)
        self.trunk_rq: faiss.ResidualQuantizer | None = None

        # Per-class tail RQs (fitted on class-specific residuals)
        self.tail_rqs: list[faiss.ResidualQuantizer] = []

        # Scaler for standardizing embeddings before RQ
        self.scaler: StandardScaler | None = None

        # Timing stats
        self.trunk_fit_time: float = 0.0
        self.tail_fit_time: float = 0.0

    def _create_rq(self, n_levels: int) -> faiss.ResidualQuantizer:
        """Create a ResidualQuantizer with proper configuration."""
        rq = faiss.ResidualQuantizer(self.embedding_dim, n_levels, self.config.nbits)
        rq.train_type = faiss.ResidualQuantizer.Train_progressive_dim
        rq.max_beam_size = self.config.beam_size
        rq.cp.niter = self.config.niter

        # Use GPU for k-means if available (single GPU is faster for small datasets)
        n_gpus = faiss.get_num_gpus()
        if self.config.use_gpu and n_gpus > 0:
            logger.info(f"    Using GPU 0 for RQ training ({n_gpus} available)")
            gpu_factory = faiss.GpuProgressiveDimIndexFactory(1)  # Use 1 GPU, not all
            rq.assign_index_factory = gpu_factory
        else:
            logger.warning(
                f"    GPU not used (use_gpu={self.config.use_gpu}, n_gpus={n_gpus})"
            )

        return rq

    def fit(self, embeddings: Tensor, labels: Tensor) -> None:
        """Fit trunk on all data, tails on per-class residuals.

        Args:
            embeddings: (N, dim) tensor of embeddings
            labels: (N,) tensor of class labels
        """
        # Convert to numpy for faiss
        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy()

        # Standardize embeddings so RQ codebook allocation is not biased
        # by feature scale
        self.scaler = StandardScaler()
        emb_np = self.scaler.fit_transform(emb_np).astype(np.float32)

        # Fit shared trunk on ALL data
        logger.info(
            f"  Fitting trunk RQ: {self.config.d_cut} levels on {len(emb_np)} samples"
        )
        trunk_start = time.time()
        self.trunk_rq = self._create_rq(self.config.d_cut)
        self.trunk_rq.train(emb_np)
        self.trunk_fit_time = time.time() - trunk_start

        # Compute trunk residuals
        trunk_codes = self.trunk_rq.compute_codes(emb_np)
        trunk_recon = self.trunk_rq.decode(trunk_codes)
        trunk_residuals = emb_np - trunk_recon

        # Fit per-class tail RQs
        logger.info(
            f"  Fitting {self.n_classes} tail RQs: {self.n_tail_levels} levels each"
        )
        tail_start = time.time()
        self.tail_rqs = []
        for c in range(self.n_classes):
            mask = labels_np == c
            class_residuals = trunk_residuals[mask]

            tail_rq = self._create_rq(self.n_tail_levels)
            tail_rq.train(class_residuals)
            self.tail_rqs.append(tail_rq)

        self.tail_fit_time = time.time() - tail_start
        logger.info(
            f"  Faiss RQ fitting complete: trunk={self.trunk_fit_time:.2f}s, "
            f"tails={self.tail_fit_time:.2f}s"
        )

    def generate_views(self, embeddings: Tensor) -> Tensor:
        """Generate m views by reconstructing through each class's tail.

        For each embedding, we encode through the shared trunk and then
        decode through each of the m class-specific tails, producing m
        different reconstructions (views).

        Args:
            embeddings: (N, dim) tensor

        Returns:
            views: (N, m, dim) tensor of m views per embedding
        """
        if self.trunk_rq is None:
            raise RuntimeError("Must call fit() before generate_views()")

        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)

        # Apply same standardization used during fit
        if self.scaler is not None:
            emb_np = self.scaler.transform(emb_np).astype(np.float32)

        # Encode through shared trunk
        trunk_codes = self.trunk_rq.compute_codes(emb_np)
        trunk_recon = self.trunk_rq.decode(trunk_codes)
        trunk_residuals = emb_np - trunk_recon

        # For each class, reconstruct through that class's tail
        views = []
        for c in range(self.n_classes):
            tail_codes = self.tail_rqs[c].compute_codes(trunk_residuals)
            tail_recon = self.tail_rqs[c].decode(tail_codes)
            full_recon = trunk_recon + tail_recon
            views.append(full_recon)

        # Stack, inverse-transform back to original space, and convert to tensor
        views_np = np.stack(views, axis=1)  # (N, m, dim)
        if self.scaler is not None:
            # Inverse transform each view
            n, m, d = views_np.shape
            views_flat = views_np.reshape(n * m, d)
            views_flat = self.scaler.inverse_transform(views_flat).astype(np.float32)
            views_np = views_flat.reshape(n, m, d)
        return torch.from_numpy(views_np).to(embeddings.device)

    def get_reconstruction_error(self, embeddings: Tensor, labels: Tensor) -> float:
        """Compute reconstruction error using correct class tails.

        Args:
            embeddings: (N, dim) tensor
            labels: (N,) tensor of true class labels

        Returns:
            Mean L2 reconstruction error
        """
        if self.trunk_rq is None:
            raise RuntimeError("Must call fit() before get_reconstruction_error()")

        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy()

        # Apply same standardization used during fit
        if self.scaler is not None:
            emb_np = self.scaler.transform(emb_np).astype(np.float32)

        # Encode through trunk
        trunk_codes = self.trunk_rq.compute_codes(emb_np)
        trunk_recon = self.trunk_rq.decode(trunk_codes)
        trunk_residuals = emb_np - trunk_recon

        # Decode through correct class tail
        full_recon = np.zeros_like(emb_np)
        for c in range(self.n_classes):
            mask = labels_np == c
            if mask.sum() > 0:
                tail_codes = self.tail_rqs[c].compute_codes(trunk_residuals[mask])
                tail_recon = self.tail_rqs[c].decode(tail_codes)
                full_recon[mask] = trunk_recon[mask] + tail_recon

        # Compute error
        error = np.linalg.norm(emb_np - full_recon, axis=1).mean()
        return float(error)


def find_optimal_d_cut(
    embeddings: Tensor,
    labels: Tensor,
    max_levels: int = 12,
    nbits: int = 6,
    use_gpu: bool = False,  # Default to CPU for d_cut detection (avoids GPU mem issues)
    divergence_threshold: float = 0.3,
) -> tuple[int, list[float]]:
    """Find trunk-tail boundary via clustering divergence.

    At each RQ level, we compare:
    - Global clustering (fitted on all data)
    - Per-class clusterings (fitted separately per class)

    When these diverge significantly, that's where the tail begins.

    NOTE: This function defaults to CPU for stability. GPU memory from faiss
    is not easily released between iterations, which can cause OOM errors.

    Args:
        embeddings: (N, dim) tensor
        labels: (N,) tensor of class labels
        max_levels: Maximum levels to test
        nbits: Bits per level
        use_gpu: Whether to use GPU (default False for stability)
        divergence_threshold: Threshold for detecting tail

    Returns:
        Tuple of (optimal_d_cut, list of divergence scores per level)
    """
    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy()
    n_classes = len(np.unique(labels_np))

    # Standardize embeddings before RQ training
    scaler = StandardScaler()
    emb_np = scaler.fit_transform(emb_np).astype(np.float32)

    residual = emb_np.copy()
    divergences: list[float] = []

    logger.info(
        f"Finding optimal d_cut (max_levels={max_levels}, use_gpu={use_gpu})..."
    )

    for level in range(max_levels):
        # Global RQ (1 level at a time) - CPU for stability
        global_rq = faiss.ResidualQuantizer(emb_np.shape[1], 1, nbits)
        global_rq.train_type = faiss.ResidualQuantizer.Train_default
        global_rq.cp.niter = 10  # Fewer iters for speed
        global_rq.train(residual)
        global_codes = global_rq.compute_codes(residual)

        # Per-class RQ at same level
        class_codes = np.zeros_like(global_codes)
        for c in range(n_classes):
            mask = labels_np == c
            if mask.sum() < 10:  # Skip tiny classes
                class_codes[mask] = global_codes[mask]
                continue

            class_rq = faiss.ResidualQuantizer(emb_np.shape[1], 1, nbits)
            class_rq.train_type = faiss.ResidualQuantizer.Train_default
            class_rq.cp.niter = 10
            class_rq.train(residual[mask])
            class_codes[mask] = class_rq.compute_codes(residual[mask])

        # Measure divergence using Adjusted Rand Index
        # ARI = 1 means identical clustering, ARI = 0 means random
        ari = adjusted_rand_score(
            global_codes.flatten().astype(int), class_codes.flatten().astype(int)
        )
        divergence = 1 - ari  # Higher = more divergent
        divergences.append(divergence)

        logger.info(f"  Level {level}: divergence={divergence:.3f} (ARI={ari:.3f})")

        # Update residual for next level
        residual = residual - global_rq.decode(global_codes)

        # Check for divergence spike (tail begins)
        if (
            level > 0
            and divergence > divergence_threshold
            and divergence > 1.5 * divergences[-2]
        ):
            logger.info(f"  -> Detected tail at level {level} (divergence spike)")
            return level, divergences

    # Default: half trunk, half tail
    default_cut = max_levels // 2
    logger.info(f"  -> No clear boundary, defaulting to d_cut={default_cut}")
    return default_cut, divergences


def compare_sklearn_vs_faiss(
    embeddings: Tensor,
    labels: Tensor,
    n_levels: int = 8,
    nbits: int = 6,
    d_cut: int = 4,
) -> dict[str, float]:
    """Benchmark faiss RQ codebook fitting.

    Returns timing for faiss RQ.
    """
    n_classes = len(torch.unique(labels))
    dim = embeddings.shape[1]

    results: dict[str, float] = {}

    # Faiss version
    logger.info("Benchmarking faiss RQ...")
    config = FaissMRQConfig(
        n_levels=n_levels,
        nbits=nbits,
        d_cut=d_cut,
        use_gpu=True,
    )
    faiss_mrq = FaissMRQModule(
        n_classes=n_classes,
        embedding_dim=dim,
        config=config,
    )
    faiss_start = time.time()
    faiss_mrq.fit(embeddings, labels)
    results["faiss_time"] = time.time() - faiss_start
    results["faiss_trunk_time"] = faiss_mrq.trunk_fit_time
    results["faiss_tail_time"] = faiss_mrq.tail_fit_time

    logger.info(
        f"Results: faiss={results['faiss_time']:.2f}s "
        f"(trunk={results['faiss_trunk_time']:.2f}s, tails={results['faiss_tail_time']:.2f}s)"
    )

    return results
