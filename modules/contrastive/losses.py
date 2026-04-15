

"""Contrastive loss functions for self-supervised and supervised learning.

This module provides:
    - info_nce_loss: InfoNCE loss for self-supervised contrastive learning
    - supcon_loss: Supervised contrastive loss using class labels

Example:
    >>> # Self-supervised
    >>> anchor = encoder(x)  # Original view
    >>> positive = augmentor.augment(x, codes)  # Augmented view
    >>> loss = info_nce_loss(anchor, positive, temperature=0.07)

    >>> # Supervised
    >>> features = encoder(x)
    >>> loss = supcon_loss(features, labels, temperature=0.07)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(
    anchor: Tensor,
    positive: Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> Tensor:
    """Compute InfoNCE loss for self-supervised contrastive learning.

    For each anchor, the positive is its augmented view. All other samples
    in the batch serve as negatives.

    Args:
        anchor: Anchor embeddings, shape (batch_size, dim).
        positive: Positive (augmented) embeddings, shape (batch_size, dim).
        temperature: Temperature scaling for softmax.
        normalize: Whether to L2-normalize embeddings before computing similarity.

    Returns:
        InfoNCE loss (scalar tensor).
    """
    if normalize:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

    batch_size = anchor.shape[0]

    representations = torch.cat([anchor, positive], dim=0)
    similarity = torch.mm(representations, representations.T) / temperature

    mask = torch.eye(2 * batch_size, device=anchor.device, dtype=torch.bool)
    similarity.masked_fill_(mask, float("-inf"))

    labels = torch.arange(batch_size, device=anchor.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    loss = F.cross_entropy(similarity, labels)

    return loss


def info_nce_loss_asymmetric(
    anchor: Tensor,
    positive: Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> Tensor:
    """Compute asymmetric InfoNCE loss (SimCLR style, one direction).

    Only computes loss for anchor -> positive direction.
    Use this when augmentation is asymmetric.

    Args:
        anchor: Anchor embeddings, shape (batch_size, dim).
        positive: Positive (augmented) embeddings, shape (batch_size, dim).
        temperature: Temperature scaling for softmax.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        InfoNCE loss (scalar tensor).
    """
    if normalize:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

    similarity = torch.mm(anchor, positive.T) / temperature

    labels = torch.arange(anchor.shape[0], device=anchor.device)

    loss = F.cross_entropy(similarity, labels)

    return loss


def supcon_loss(
    features: Tensor,
    labels: Tensor,
    temperature: float = 0.07,
    base_temperature: float = 0.07,
    normalize: bool = True,
) -> Tensor:
    """Compute Supervised Contrastive loss (SupCon).

    Samples with the same label are positives; different labels are negatives.
    This is the loss from "Supervised Contrastive Learning" (Khosla et al., 2020).

    Args:
        features: Embeddings, shape (batch_size, dim).
        labels: Class labels, shape (batch_size,).
        temperature: Temperature for similarity scaling.
        base_temperature: Base temperature for loss scaling.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        SupCon loss (scalar tensor).
    """
    if normalize:
        features = F.normalize(features, dim=-1)

    batch_size = features.shape[0]
    device = features.device

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    similarity = torch.mm(features, features.T) / temperature

    logits_mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
    mask = mask * logits_mask.float()

    self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    similarity.masked_fill_(self_mask, float("-inf"))

    exp_logits = torch.exp(similarity) * logits_mask.float()
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    positives_per_sample = mask.sum(dim=1)
    has_positives = positives_per_sample > 0

    if not has_positives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives_per_sample + 1e-8)

    loss = -(temperature / base_temperature) * mean_log_prob_pos[has_positives].mean()

    return loss


def trunk_consistency_loss(
    anchor_residual: Tensor,
    positive_residual: Tensor,
    anchor_codes: Tensor,
    positive_codes: Tensor,
    codebooks: list[Tensor],
    d_cut: int,
) -> Tensor:
    """Compute trunk consistency loss.

    Encourages augmented views to stay close to anchor's trunk centroids.
    This is L_trunk from the paper.

    Args:
        anchor_residual: Anchor's residual after each trunk level.
        positive_residual: Positive's residual after each trunk level.
        anchor_codes: Anchor's codes, shape (batch_size, n_levels).
        positive_codes: Positive's codes, shape (batch_size, n_levels).
        codebooks: List of codebook tensors, one per level.
        d_cut: Trunk-tail cutoff level.

    Returns:
        Trunk consistency loss (scalar tensor).
    """
    loss = torch.tensor(0.0, device=anchor_residual.device)

    for level in range(d_cut):
        codebook = codebooks[level]
        anchor_centroid = codebook[anchor_codes[:, level]]

        dist = (positive_residual - anchor_centroid).pow(2).sum(dim=-1).mean()
        loss = loss + dist

    return loss


def tail_divergence_loss(
    anchor_codes: Tensor,
    positive_codes: Tensor,
    d_cut: int,
    n_levels: int,
) -> Tensor:
    """Compute tail divergence loss.

    Encourages augmented views to use different tail codes than anchor.
    This is L_tail from the paper (negative of collision count).

    Args:
        anchor_codes: Anchor's codes, shape (batch_size, n_levels).
        positive_codes: Positive's codes, shape (batch_size, n_levels).
        d_cut: Trunk-tail cutoff level.
        n_levels: Total number of levels.

    Returns:
        Tail divergence loss (scalar tensor). Lower is better (more divergence).
    """
    tail_anchor = anchor_codes[:, d_cut:n_levels]
    tail_positive = positive_codes[:, d_cut:n_levels]

    collisions = (tail_anchor == tail_positive).float().mean()

    return collisions


def supcon_proto_loss(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    temperature: float = 0.07,
    prototype_weight: float = 0.5,
    normalize: bool = True,
) -> Tensor:
    """Supervised contrastive loss with class prototype regularization.

    Combines SupCon (same-label samples as positives) with an attraction
    term that pulls each sample toward its class prototype centroid.
    Prototypes are derived from RQ codebook centroids (trunk reconstruction
    centroids per class).

    Args:
        features: Projected embeddings, shape (batch_size, dim).
        labels: Class labels, shape (batch_size,).
        prototypes: Per-class prototype vectors, shape (n_classes, dim).
        temperature: Temperature for similarity scaling.
        prototype_weight: Weight for the prototype attraction term.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        Combined loss (scalar tensor).
    """
    supcon = supcon_loss(features, labels, temperature=temperature, normalize=normalize)

    if normalize:
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)

    # Pull each sample toward its class prototype
    proto_for_sample = prototypes[labels]  # (batch_size, dim)
    proto_sim = (features * proto_for_sample).sum(dim=-1) / temperature
    proto_loss = -proto_sim.mean()

    return supcon + prototype_weight * proto_loss


def weighted_supcon_loss(
    features: Tensor,
    labels: Tensor,
    recon_quality: Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> Tensor:
    """Weighted supervised contrastive loss using RQ reconstruction quality.

    Each positive pair's contribution is weighted by how well the RQ
    reconstructs the sample. Better-reconstructed samples are more
    reliable anchors, so they get higher weight.

    Args:
        features: Projected embeddings, shape (batch_size, dim).
        labels: Class labels, shape (batch_size,).
        recon_quality: Per-sample reconstruction quality scores in [0, 1],
            shape (batch_size,). Higher = better reconstruction.
        temperature: Temperature for similarity scaling.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        Weighted SupCon loss (scalar tensor).
    """
    if normalize:
        features = F.normalize(features, dim=-1)

    batch_size = features.shape[0]
    device = features.device

    labels_col = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels_col, labels_col.T).float().to(device)

    similarity = torch.mm(features, features.T) / temperature

    logits_mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
    mask = mask * logits_mask.float()

    self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    similarity.masked_fill_(self_mask, float("-inf"))

    exp_logits = torch.exp(similarity) * logits_mask.float()
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Weight by reconstruction quality of the positive
    weights = recon_quality.view(1, -1).expand(batch_size, -1)  # (B, B)
    weighted_mask = mask * weights

    positives_per_sample = weighted_mask.sum(dim=1)
    has_positives = positives_per_sample > 0

    if not has_positives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (weighted_mask * log_prob).sum(dim=1) / (
        positives_per_sample + 1e-8
    )

    loss = -mean_log_prob_pos[has_positives].mean()

    return loss


def hard_neg_supcon_loss(
    features: Tensor,
    labels: Tensor,
    temperature: float = 0.07,
    hard_neg_strength: float = 0.5,
    normalize: bool = True,
) -> Tensor:
    """Supervised contrastive loss with hard negative mining.

    Up-weights negative samples that are close to the anchor in
    embedding space. These hard negatives are more informative for
    learning discriminative representations.

    Args:
        features: Projected embeddings, shape (batch_size, dim).
        labels: Class labels, shape (batch_size,).
        temperature: Temperature for similarity scaling.
        hard_neg_strength: Controls how much to up-weight hard negatives.
            0 = standard SupCon, 1 = fully hard-negative weighted.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        Hard-negative SupCon loss (scalar tensor).
    """
    if normalize:
        features = F.normalize(features, dim=-1)

    batch_size = features.shape[0]
    device = features.device

    labels_col = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels_col, labels_col.T).float().to(device)
    neg_mask = 1.0 - pos_mask

    similarity = torch.mm(features, features.T) / temperature

    logits_mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
    pos_mask = pos_mask * logits_mask.float()

    # Up-weight hard negatives (high similarity but different label)
    neg_weights = neg_mask * logits_mask.float()
    sim_detached = similarity.detach()
    # Softmax over negatives to concentrate on hardest ones
    hard_neg_weights = neg_weights * torch.exp(hard_neg_strength * sim_detached)
    hard_neg_weights = hard_neg_weights / (
        hard_neg_weights.sum(dim=1, keepdim=True) + 1e-8
    )
    # Combine: all negatives contribute, but hard ones contribute more
    combined_neg_weights = (1 - hard_neg_strength) * neg_weights / (
        neg_weights.sum(dim=1, keepdim=True) + 1e-8
    ) + (hard_neg_strength) * hard_neg_weights

    self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    similarity.masked_fill_(self_mask, float("-inf"))

    # Denominator: re-weighted negatives + positives
    exp_sim = torch.exp(similarity)
    denom = (exp_sim * combined_neg_weights).sum(dim=1, keepdim=True) * (
        neg_weights.sum(dim=1, keepdim=True)
    ) + (exp_sim * pos_mask).sum(dim=1, keepdim=True)

    log_prob = similarity - torch.log(denom + 1e-8)

    positives_per_sample = pos_mask.sum(dim=1)
    has_positives = positives_per_sample > 0

    if not has_positives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (positives_per_sample + 1e-8)

    loss = -mean_log_prob_pos[has_positives].mean()

    return loss
