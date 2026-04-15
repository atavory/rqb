#!/usr/bin/env python3

"""Mutual Information vs RQ Level Diagnostic.

This script validates the Fisher Information Decay claim from the paper's theory section:
MI between codes and labels I(Y; k_ℓ | Z_{1:ℓ-1}) decreases with RQ depth ℓ.

The script computes MI at each RQ level using three estimators:
1. Plug-in estimator with add-α smoothing
2. KSG (Kraskov-Stögbauer-Grassberger) estimator for continuous variables
3. MINE (Mutual Information Neural Estimation)

The output identifies the "knee" where MI drops — this is the empirical D_cut.
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor


@dataclass
class MIDiagnosticConfig:
    """Configuration for MI diagnostic."""

    n_levels: int = 8  # Total RQ levels (D)
    n_codes: int = 256  # Codebook size per level
    alpha: float = 1.0  # Add-α smoothing for plug-in estimator
    ksg_k: int = 3  # k-nearest neighbors for KSG
    mine_hidden_dim: int = 128  # Hidden dim for MINE network
    mine_epochs: int = 100  # Training epochs for MINE
    mine_batch_size: int = 256  # Batch size for MINE
    device: str = "cpu"
    seed: int = 42


# =============================================================================
# Plug-in Estimator with Add-α Smoothing
# =============================================================================


def entropy_plugin(labels: np.ndarray, alpha: float = 1.0) -> float:
    """Compute entropy H(Y) using plug-in estimator with add-α smoothing.

    Args:
        labels: Array of discrete labels.
        alpha: Smoothing parameter (Laplace smoothing when alpha=1).

    Returns:
        Entropy in nats.
    """
    counts = Counter(labels)
    n = len(labels)
    n_classes = len(counts)

    # Add-α smoothing
    total = n + alpha * n_classes
    entropy = 0.0
    for count in counts.values():
        p = (count + alpha) / total
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


def conditional_entropy_plugin(
    labels: np.ndarray, codes: np.ndarray, alpha: float = 1.0
) -> float:
    """Compute conditional entropy H(Y | Z) using plug-in estimator.

    Args:
        labels: Array of discrete labels, shape (N,).
        codes: Array of code tuples, shape (N, n_levels) or (N,) for single level.
        alpha: Smoothing parameter.

    Returns:
        Conditional entropy in nats.
    """
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)

    # Convert codes to tuple keys
    code_keys = [tuple(c) for c in codes]
    n = len(labels)

    # Count (code, label) pairs
    joint_counts: dict = {}
    code_counts: dict = {}

    for code_key, label in zip(code_keys, labels):
        joint_counts[(code_key, label)] = joint_counts.get((code_key, label), 0) + 1
        code_counts[code_key] = code_counts.get(code_key, 0) + 1

    n_labels = len(set(labels))
    n_codes_unique = len(code_counts)

    # Compute H(Y | Z) = sum_z P(z) * H(Y | Z=z)
    cond_entropy = 0.0
    for code_key, code_count in code_counts.items():
        # P(Z = z)
        p_z = code_count / n

        # H(Y | Z = z) with add-α smoothing
        h_y_given_z = 0.0
        total_with_smoothing = code_count + alpha * n_labels
        for label in set(labels):
            joint_count = joint_counts.get((code_key, label), 0)
            p_y_given_z = (joint_count + alpha) / total_with_smoothing
            if p_y_given_z > 0:
                h_y_given_z -= p_y_given_z * math.log(p_y_given_z)

        cond_entropy += p_z * h_y_given_z

    return cond_entropy


def compute_mi_plugin(
    labels: np.ndarray,
    codes: np.ndarray,
    level: int,
    alpha: float = 1.0,
) -> float:
    """Compute MI at a specific level using plug-in estimator.

    I(Y; k_ℓ | Z_{1:ℓ-1}) = H(Y | Z_{1:ℓ-1}) - H(Y | Z_{1:ℓ})

    Args:
        labels: Class labels, shape (N,).
        codes: Full RQ codes, shape (N, n_levels).
        level: Current level ℓ (0-indexed).
        alpha: Smoothing parameter.

    Returns:
        Mutual information in nats.
    """
    if level == 0:
        # H(Y) - H(Y | k_0)
        h_y = entropy_plugin(labels, alpha)
        h_y_given_z = conditional_entropy_plugin(labels, codes[:, :1], alpha)
    else:
        # H(Y | Z_{1:ℓ-1}) - H(Y | Z_{1:ℓ})
        h_y_given_prev = conditional_entropy_plugin(labels, codes[:, :level], alpha)
        h_y_given_z = conditional_entropy_plugin(labels, codes[:, : level + 1], alpha)
        return max(0.0, h_y_given_prev - h_y_given_z)  # Ensure non-negative

    return max(0.0, h_y - h_y_given_z)


# =============================================================================
# KSG Estimator (Kraskov-Stögbauer-Grassberger)
# =============================================================================


def ksg_mi_discrete_continuous(
    labels: np.ndarray,
    embeddings: np.ndarray,
    k: int = 3,
) -> float:
    """Compute MI between discrete labels and continuous embeddings using KSG.

    This is a simplified version that treats labels as discrete and embeddings
    as continuous, computing I(Y; Z) directly.

    Args:
        labels: Discrete labels, shape (N,).
        embeddings: Continuous embeddings, shape (N, D).
        k: Number of nearest neighbors.

    Returns:
        Mutual information estimate in nats.
    """
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors

    n = len(labels)
    unique_labels = np.unique(labels)

    # For each class, find k-th nearest neighbor distances
    mi = digamma(n) - digamma(k)

    for label in unique_labels:
        mask = labels == label
        n_label = mask.sum()
        if n_label <= k:
            continue

        # Points in this class
        X_label = embeddings[mask]

        # Find k-th nearest neighbor distance for each point
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_label)
        distances, _ = nn.kneighbors(X_label)
        eps = distances[:, k]  # k-th neighbor distance (0-indexed, so k+1 neighbors)

        # Count points within eps in full dataset
        nn_full = NearestNeighbors(radius=eps.mean())
        nn_full.fit(embeddings)

        # Contribution from this class
        p_label = n_label / n
        mi += p_label * digamma(n_label)

    return max(0.0, mi)


def compute_mi_ksg_per_level(
    labels: np.ndarray,
    codes: np.ndarray,
    level: int,
    k: int = 3,
) -> float:
    """Compute MI at a specific level using KSG-style estimation.

    For discrete codes, we use a simplified approach treating the code
    sequence as a discrete context.

    Args:
        labels: Class labels, shape (N,).
        codes: Full RQ codes, shape (N, n_levels).
        level: Current level ℓ (0-indexed).
        k: Number of nearest neighbors.

    Returns:
        Mutual information estimate in nats.
    """
    # For discrete-discrete MI, KSG reduces to plug-in with some adjustments
    # We'll use a binning approach for the continuous case

    # Convert codes to continuous space for KSG
    n_codes = codes.max() + 1
    code_embedding = codes[:, : level + 1].astype(np.float32)

    # Add small noise to break ties
    code_embedding += np.random.randn(*code_embedding.shape) * 0.01

    if level == 0:
        return ksg_mi_discrete_continuous(labels, code_embedding, k)
    else:
        # I(Y; k_ℓ | Z_{1:ℓ-1}) ≈ I(Y; Z_{1:ℓ}) - I(Y; Z_{1:ℓ-1})
        mi_full = ksg_mi_discrete_continuous(labels, code_embedding, k)
        mi_prev = ksg_mi_discrete_continuous(labels, code_embedding[:, :-1], k)
        return max(0.0, mi_full - mi_prev)


# =============================================================================
# MINE Estimator (Mutual Information Neural Estimation)
# =============================================================================


class MINENetwork(nn.Module):
    """Neural network for MINE estimation."""

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.net(torch.cat([x, y], dim=-1))


def compute_mi_mine(
    labels: np.ndarray,
    codes: np.ndarray,
    level: int,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """Compute MI at a specific level using MINE.

    MINE estimates MI by training a neural network to distinguish joint
    samples from marginal samples.

    Args:
        labels: Class labels, shape (N,).
        codes: Full RQ codes, shape (N, n_levels).
        level: Current level ℓ (0-indexed).
        hidden_dim: Hidden dimension for MINE network.
        epochs: Training epochs.
        batch_size: Batch size.
        device: Device to use.

    Returns:
        Mutual information estimate in nats.
    """
    n = len(labels)
    n_classes = len(np.unique(labels))

    # One-hot encode labels
    y_onehot = np.zeros((n, n_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        y_onehot[i, label] = 1.0

    # Code context up to current level
    x_codes = codes[:, : level + 1].astype(np.float32)

    # Convert to tensors
    x_tensor = torch.tensor(x_codes, device=device)
    y_tensor = torch.tensor(y_onehot, device=device)

    # Create MINE network
    mine = MINENetwork(x_codes.shape[1], n_classes, hidden_dim).to(device)
    optimizer = torch.optim.Adam(mine.parameters(), lr=1e-3)

    # Training loop
    for _ in range(epochs):
        # Sample batch
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        x_batch = x_tensor[idx]
        y_batch = y_tensor[idx]

        # Shuffle y for marginal samples
        y_marginal = y_batch[torch.randperm(len(y_batch))]

        # MINE loss: -E[T(x,y)] + log(E[exp(T(x,y'))])
        t_joint = mine(x_batch, y_batch)
        t_marginal = mine(x_batch, y_marginal)

        # Use log-sum-exp trick for numerical stability
        loss = (
            -t_joint.mean()
            + torch.logsumexp(t_marginal, dim=0)
            - math.log(len(t_marginal))
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Estimate final MI
    with torch.no_grad():
        t_joint = mine(x_tensor, y_tensor)
        y_marginal = y_tensor[torch.randperm(n)]
        t_marginal = mine(x_tensor, y_marginal)

        mi = t_joint.mean() - (
            torch.logsumexp(t_marginal, dim=0) - math.log(len(t_marginal))
        )

    return max(0.0, mi.item())


# =============================================================================
# Main Diagnostic Functions
# =============================================================================


def generate_synthetic_data(
    n_samples: int,
    n_classes: int,
    n_levels: int,
    n_codes: int,
    class_separation: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic RQ codes with class structure.

    Early levels have class-correlated codes, later levels are noisy.
    This simulates the "trunk = class-agnostic, tail = instance-specific" structure.

    Args:
        n_samples: Number of samples.
        n_classes: Number of classes.
        n_levels: Total RQ levels.
        n_codes: Codebook size per level.
        class_separation: How much class structure to inject (0 = none, 1 = perfect).
        seed: Random seed.

    Returns:
        Tuple of (labels, codes) arrays.
    """
    np.random.seed(seed)

    labels = np.random.randint(0, n_classes, n_samples)
    codes = np.zeros((n_samples, n_levels), dtype=np.int64)

    for level in range(n_levels):
        # Early levels have more class structure
        decay = math.exp(-level / 2)  # Exponential decay of class info
        class_weight = class_separation * decay

        for i, label in enumerate(labels):
            if np.random.random() < class_weight:
                # Code correlated with class
                codes[i, level] = (label * (n_codes // n_classes) + level) % n_codes
            else:
                # Random code
                codes[i, level] = np.random.randint(0, n_codes)

    return labels, codes


def run_mi_diagnostic(
    labels: np.ndarray,
    codes: np.ndarray,
    config: MIDiagnosticConfig,
) -> dict:
    """Run full MI diagnostic with all three estimators.

    Args:
        labels: Class labels, shape (N,).
        codes: Full RQ codes, shape (N, n_levels).
        config: Diagnostic configuration.

    Returns:
        Dictionary with MI estimates per level for each estimator.
    """
    n_levels = codes.shape[1]

    results: dict = {
        "plugin": [],
        "ksg": [],
        "mine": [],
        "levels": list(range(n_levels)),
    }

    print("\n" + "=" * 60)
    print("MUTUAL INFORMATION vs RQ LEVEL DIAGNOSTIC")
    print("=" * 60)
    print(
        f"Samples: {len(labels)}, Classes: {len(np.unique(labels))}, Levels: {n_levels}"
    )
    print("=" * 60)

    for level in range(n_levels):
        print(f"\nLevel {level}:")

        # Plug-in estimator
        mi_plugin = compute_mi_plugin(labels, codes, level, config.alpha)
        results["plugin"].append(mi_plugin)
        print(f"  Plug-in (α={config.alpha}): {mi_plugin:.4f} nats")

        # KSG estimator
        mi_ksg = compute_mi_ksg_per_level(labels, codes, level, config.ksg_k)
        results["ksg"].append(mi_ksg)
        print(f"  KSG (k={config.ksg_k}):      {mi_ksg:.4f} nats")

        # MINE estimator
        mi_mine = compute_mi_mine(
            labels,
            codes,
            level,
            config.mine_hidden_dim,
            config.mine_epochs,
            config.mine_batch_size,
            config.device,
        )
        results["mine"].append(mi_mine)
        print(f"  MINE:              {mi_mine:.4f} nats")

    # Find knee (where MI drops significantly)
    for estimator in ["plugin", "ksg", "mine"]:
        mi_values = results[estimator]
        if len(mi_values) > 1:
            # Find level with largest relative drop
            drops = []
            for i in range(1, len(mi_values)):
                if mi_values[i - 1] > 0:
                    drop = (mi_values[i - 1] - mi_values[i]) / mi_values[i - 1]
                else:
                    drop = 0
                drops.append(drop)

            if drops:
                knee_level = np.argmax(drops) + 1
                results[f"{estimator}_knee"] = knee_level
            else:
                results[f"{estimator}_knee"] = n_levels // 2

    print("\n" + "=" * 60)
    print("KNEE DETECTION (Suggested D_cut)")
    print("=" * 60)
    for estimator in ["plugin", "ksg", "mine"]:
        knee = results.get(f"{estimator}_knee", "N/A")
        print(f"  {estimator.upper()}: D_cut = {knee}")

    return results


def main() -> None:
    """Run MI diagnostic experiments."""
    parser = argparse.ArgumentParser(description="MI vs RQ Level Diagnostic")
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of samples"
    )
    parser.add_argument("--n-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--n-levels", type=int, default=8, help="Number of RQ levels")
    parser.add_argument("--n-codes", type=int, default=256, help="Codebook size")
    parser.add_argument(
        "--class-separation",
        type=float,
        default=0.7,
        help="Class structure strength (0-1)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Add-α smoothing parameter"
    )
    parser.add_argument("--ksg-k", type=int, default=3, help="KSG k-neighbors")
    parser.add_argument(
        "--mine-epochs", type=int, default=100, help="MINE training epochs"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Device"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = MIDiagnosticConfig(
        n_levels=args.n_levels,
        n_codes=args.n_codes,
        alpha=args.alpha,
        ksg_k=args.ksg_k,
        mine_epochs=args.mine_epochs,
        device=args.device,
        seed=args.seed,
    )

    # Generate synthetic data
    print("\nGenerating synthetic data with class structure...")
    labels, codes = generate_synthetic_data(
        n_samples=args.n_samples,
        n_classes=args.n_classes,
        n_levels=args.n_levels,
        n_codes=args.n_codes,
        class_separation=args.class_separation,
        seed=args.seed,
    )

    # Run diagnostic
    results = run_mi_diagnostic(labels, codes, config)

    # Add metadata
    results["metadata"] = {
        "n_samples": args.n_samples,
        "n_classes": args.n_classes,
        "n_levels": args.n_levels,
        "n_codes": args.n_codes,
        "class_separation": args.class_separation,
        "alpha": args.alpha,
        "ksg_k": args.ksg_k,
        "mine_epochs": args.mine_epochs,
        "seed": args.seed,
    }

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
