

"""SCARF: Self-supervised Contrastive Learning using Random Feature corruption.

Implementation based on "SCARF: Self-Supervised Contrastive Learning using
Random Feature Corruption" (Bahri et al., ICML 2022).

SCARF produces general-purpose tabular embeddings without labels:
    1. Corrupt a random subset of features (replace with marginal distribution)
    2. InfoNCE contrastive loss between original and corrupted views
    3. Frozen encoder produces embeddings for downstream RQ analysis

This supports the paper's "frozen pre-trained embeddings" narrative without
the circularity of supervised encoders (which need labels to train, then
"discover" class structure from labels).

Example:
    >>> pretrainer = SCARFPretrainer(
    ...     input_dim=14,
    ...     embedding_dim=64,
    ...     hidden_dims=[128, 128],
    ... )
    >>> encoder = pretrainer.fit(features_np, epochs=50)
    >>> embeddings = pretrainer.transform(features_np)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SCARFEncoder(nn.Module):
    """MLP encoder for SCARF self-supervised pre-training.

    Takes raw concatenated features (continuous + categorical cast to float)
    and produces a fixed-dim embedding. Architecture matches MLPEncoder but
    with a simpler interface (single input tensor, no cat/cont split).

    Attributes:
        input_dim: Number of input features.
        embedding_dim: Output embedding dimension.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize SCARF encoder.

        Args:
            input_dim: Number of input features (d_raw).
            embedding_dim: Output embedding dimension.
            hidden_dims: Hidden layer dimensions. Defaults to [128, 128].
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Raw features, shape (batch_size, input_dim).

        Returns:
            Embeddings, shape (batch_size, embedding_dim).
        """
        return self.mlp(self.input_bn(x))


class SCARFPretrainer:
    """SCARF pre-trainer: trains a self-supervised encoder on tabular data.

    Corruption strategy: for each sample in a batch, randomly select
    `corruption_rate` fraction of features and replace them with values
    drawn from that feature's empirical marginal distribution (computed
    once from the training set).

    Training uses InfoNCE loss between the embedding of the original
    sample and its corrupted view, with all other batch samples as
    negatives.

    Example:
        >>> pretrainer = SCARFPretrainer(input_dim=14, embedding_dim=64)
        >>> encoder = pretrainer.fit(features_np, epochs=50, device="cuda")
        >>> embeddings = pretrainer.transform(features_np)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        corruption_rate: float = 0.3,
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
    ) -> None:
        """Initialize SCARF pre-trainer.

        Args:
            input_dim: Number of input features.
            embedding_dim: Output embedding dimension.
            hidden_dims: Hidden layer dimensions for encoder MLP.
            corruption_rate: Fraction of features to corrupt per sample.
            temperature: InfoNCE temperature parameter.
            lr: Learning rate for AdamW.
            weight_decay: Weight decay for AdamW.
            batch_size: Training batch size.
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.corruption_rate = corruption_rate
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.encoder: SCARFEncoder | None = None
        # Per-feature marginal values for corruption, shape (N, d)
        self._marginal_pool: np.ndarray | None = None

    def _corrupt(
        self,
        x: Tensor,
        rng: torch.Generator,
    ) -> Tensor:
        """Create a corrupted view by replacing random features with marginals.

        Args:
            x: Original features, shape (batch_size, input_dim).
            rng: Torch random generator for reproducibility.

        Returns:
            Corrupted features, shape (batch_size, input_dim).
        """
        batch_size = x.shape[0]
        device = x.device

        # Binary mask: 1 = corrupt this feature
        # Generate on CPU (matching the generator device), then move to target device
        mask = (
            torch.rand(batch_size, self.input_dim, generator=rng) < self.corruption_rate
        ).to(device)

        # Sample replacement values from the marginal pool
        assert self._marginal_pool is not None
        pool = self._marginal_pool
        pool_size = pool.shape[0]
        # Random indices into the pool for each (sample, feature) pair
        idx = torch.randint(
            0, pool_size, (batch_size,), generator=rng, device=torch.device("cpu")
        )
        replacements = torch.from_numpy(pool[idx.numpy()]).to(device)

        # Apply corruption
        corrupted = x.clone()
        corrupted[mask] = replacements[mask]
        return corrupted

    def fit(
        self,
        features: np.ndarray,
        epochs: int = 50,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ) -> SCARFEncoder:
        """Pre-train the SCARF encoder on raw features.

        Args:
            features: (N, d) float32 feature matrix. Labels are NOT used.
            epochs: Number of training epochs.
            device: Device string ("cpu" or "cuda").
            seed: Random seed for reproducibility.
            verbose: Whether to print training progress.

        Returns:
            Frozen SCARFEncoder (eval mode, requires_grad=False).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        device_obj = torch.device(device)

        # Store the full training set as the marginal pool for corruption
        self._marginal_pool = features.copy()

        # Build encoder
        encoder = SCARFEncoder(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
        ).to(device_obj)

        optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        rng = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        # Convert to tensor dataset
        features_t = torch.from_numpy(features).float()
        dataset = torch.utils.data.TensorDataset(features_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            drop_last=True,  # InfoNCE needs consistent batch sizes
        )

        encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                batch_x = batch_x.to(device_obj)

                # Create corrupted view
                corrupted_x = self._corrupt(batch_x, rng)

                # Encode both views
                z_orig = encoder(batch_x)
                z_corrupt = encoder(corrupted_x)

                # InfoNCE loss
                loss = self._info_nce(z_orig, z_corrupt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(
                    f"  SCARF epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}",
                    flush=True,
                )

        # Freeze encoder
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        self.encoder = encoder
        return encoder

    def _info_nce(self, anchor: Tensor, positive: Tensor) -> Tensor:
        """Compute InfoNCE loss between anchor and positive views.

        Args:
            anchor: Original embeddings, shape (B, d).
            positive: Corrupted embeddings, shape (B, d).

        Returns:
            InfoNCE loss (scalar).
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # Similarity matrix: anchor_i vs positive_j
        logits = torch.mm(anchor, positive.T) / self.temperature
        # Positive pairs are on the diagonal
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return F.cross_entropy(logits, labels)

    def transform(
        self,
        features: np.ndarray,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Extract embeddings from a frozen SCARF encoder.

        Args:
            features: (N, d) float32 feature matrix.
            device: Device for inference.
            batch_size: Batch size for inference.

        Returns:
            (N, embedding_dim) float32 embedding matrix.
        """
        assert self.encoder is not None, "Must call fit() before transform()"
        encoder = self.encoder
        device_obj = torch.device(device)
        encoder = encoder.to(device_obj)

        features_t = torch.from_numpy(features).float()
        all_emb: list[Tensor] = []

        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(features_t), batch_size):
                batch = features_t[i : i + batch_size].to(device_obj)
                emb = encoder(batch)
                all_emb.append(emb.cpu())

        return torch.cat(all_emb).numpy().astype(np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save encoder weights and architecture config to a checkpoint.

        The checkpoint contains everything needed to reconstruct and load
        the encoder without knowing the original training config:
            - encoder state_dict (model weights)
            - architecture config (input_dim, embedding_dim, hidden_dims)

        Args:
            path: File path for the checkpoint (.pt).
        """
        assert self.encoder is not None, "Must call fit() before save_checkpoint()"
        encoder = self.encoder
        checkpoint = {
            "encoder_state_dict": encoder.state_dict(),
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "SCARFPretrainer":
        """Load a pre-trained SCARF encoder from a checkpoint.

        Reconstructs the encoder architecture from the saved config and
        loads the weights. The returned pretrainer has a frozen encoder
        ready for `transform()`.

        Args:
            path: File path to the checkpoint (.pt).
            device: Device to load the encoder onto.

        Returns:
            SCARFPretrainer with a frozen encoder loaded from checkpoint.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        input_dim: int = checkpoint["input_dim"]
        embedding_dim: int = checkpoint["embedding_dim"]
        hidden_dims: list[int] | None = checkpoint["hidden_dims"]

        pretrainer = cls(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
        )

        encoder = SCARFEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
        )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder = encoder.to(torch.device(device))

        pretrainer.encoder = encoder
        return pretrainer
