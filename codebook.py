#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""RQ codebook training and encoding for trunk-tail bandits."""

from __future__ import annotations

import numpy as np


def train_rq_codebook(
    features: np.ndarray,
    d_cut: int,
    nbits: int = 4,
) -> tuple[object, list[np.ndarray]]:
    """Train a Residual Quantizer codebook on features.

    Args:
        features: (N, dim) float32 array of feature vectors.
        d_cut: Number of RQ levels (depth).
        nbits: Bits per level (b = 2^nbits centroids per level).

    Returns:
        rq: Trained faiss.ResidualQuantizer.
        centroids: List of (b, dim) arrays, one per level.
    """
    import faiss

    features = np.ascontiguousarray(features, dtype=np.float32)
    dim = features.shape[1]

    rq = faiss.ResidualQuantizer(dim, d_cut, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.verbose = False
    rq.train(features)

    b = 1 << nbits
    cb_flat = faiss.vector_to_array(rq.codebooks).reshape(d_cut, b, dim)
    centroids = [cb_flat[level].copy() for level in range(d_cut)]

    return rq, centroids


def encode(
    rq: object,
    features: np.ndarray,
    d_cut: int,
    nbits: int = 4,
) -> np.ndarray:
    """Encode features into per-level integer codes using a trained RQ.

    Args:
        rq: Trained faiss.ResidualQuantizer.
        features: (N, dim) float32 array.
        d_cut: Number of RQ levels.
        nbits: Bits per level.

    Returns:
        codes: (N, d_cut) int64 array of per-level code assignments.
    """
    features = np.ascontiguousarray(features, dtype=np.float32)
    n_samples = features.shape[0]

    packed_bytes = rq.compute_codes(features)
    mask = (1 << nbits) - 1
    packed_ints = np.zeros(n_samples, dtype=np.int64)
    for byte_idx in range(packed_bytes.shape[1]):
        packed_ints += packed_bytes[:, byte_idx].astype(np.int64) << (8 * byte_idx)

    codes = np.zeros((n_samples, d_cut), dtype=np.int64)
    for level in range(d_cut):
        codes[:, level] = (packed_ints >> (level * nbits)) & mask

    return codes


def compute_residual_features(
    features: np.ndarray,
    codes: np.ndarray,
    centroids: list[np.ndarray],
    d_cut: int,
) -> np.ndarray:
    """Compute per-level residual features for DRQm bandits.

    residual[i, level] = features[i] - sum(centroids[j][codes[i,j]] for j < level)

    Args:
        features: (N, dim) array.
        codes: (N, d_cut) int64 array of per-level codes.
        centroids: List of (b, dim) arrays per level.
        d_cut: Number of levels.

    Returns:
        residuals: (N, d_cut, dim) array of residual features.
    """
    n_samples, dim = features.shape
    residuals = np.zeros((n_samples, d_cut, dim), dtype=np.float32)
    cumulative = np.zeros((n_samples, dim), dtype=np.float32)

    for level in range(d_cut):
        residuals[:, level, :] = features - cumulative
        for i in range(n_samples):
            cumulative[i] += centroids[level][codes[i, level]]

    return residuals
