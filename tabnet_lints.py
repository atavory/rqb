# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""TabNet-LinTS contextual bandit.

Unsupervised TabNet pretraining on the codebook samples (same data used for
RQ codebook training), producing 32-dim representations.  LinTS then runs
on these learned features.

This is a strong neural baseline: TabNet's sequential attention mechanism
learns feature interactions without labels, and LinTS provides calibrated
Thompson Sampling exploration on the compressed representation.

Reference: Arik & Pfister, "TabNet: Attentive Interpretable Tabular
Learning" (AAAI 2021).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

logger: logging.Logger = logging.getLogger(__name__)


class TabNetEncoder(torch.nn.Module):
    """Minimal TabNet encoder for unsupervised pretraining.

    Single-step attention (N_steps=1) for speed.  Outputs a fixed-size
    representation suitable for downstream LinTS.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        n_d: int = 32,
        n_a: int = 32,
        relaxation_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initial batch norm
        self.initial_bn = torch.nn.BatchNorm1d(input_dim)

        # Shared and step-specific layers
        self.shared_fc = torch.nn.Linear(input_dim, n_d + n_a, bias=False)
        self.shared_bn = torch.nn.BatchNorm1d(n_d + n_a)

        # Attention transformer
        self.attention_fc = torch.nn.Linear(n_a, input_dim, bias=False)
        self.attention_bn = torch.nn.BatchNorm1d(input_dim)

        # Output projection
        self.output_fc = torch.nn.Linear(n_d, output_dim)

        self.n_d = n_d
        self.n_a = n_a
        self.relaxation_factor = relaxation_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to output_dim features."""
        bs = x.shape[0]
        x = self.initial_bn(x)

        # Prior scales (uniform initially)
        prior_scales = torch.ones(bs, self.input_dim, device=x.device)

        # Attention: softmax over features weighted by prior
        h = self.shared_fc(x * prior_scales)
        h = self.shared_bn(h)
        h = torch.relu(h)

        # Split into decision (n_d) and attention (n_a) parts
        h_d = h[:, : self.n_d]
        h_a = h[:, self.n_d :]

        # Compute attention mask (not used for gating in single-step,
        # but keeps architecture consistent)
        _attn = self.attention_fc(h_a)
        _attn = self.attention_bn(_attn)

        # Output
        out = self.output_fc(torch.relu(h_d))
        return out


class TabNetDecoder(torch.nn.Module):
    """Decoder for TabNet unsupervised pretraining (reconstruction)."""

    def __init__(self, output_dim: int, input_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(output_dim, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(output_dim)
        self.fc2 = torch.nn.Linear(output_dim, input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.fc1(z)))
        return self.fc2(h)


class TabNetLinTS:
    """TabNet unsupervised pretraining + LinTS on learned features.

    Workflow:
    1. Pretrain TabNet encoder-decoder on codebook samples (reconstruction).
    2. Encode all eval samples to ``output_dim``-dimensional features.
    3. Run standard LinTS on encoded features.

    The caller is responsible for providing codebook data at init time.
    """

    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        codebook_data: np.ndarray,
        output_dim: int = 32,
        pretrain_epochs: int = 50,
        pretrain_lr: float = 1e-3,
        pretrain_batch_size: int = 256,
        lambda_prior: float = 1.0,
        nu: float = 0.1,
        rng: np.random.RandomState | None = None,
        mask_fraction: float = 0.3,
    ) -> None:
        self.n_arms = n_arms
        self.output_dim = output_dim
        self.rng = rng or np.random.RandomState()
        self.device = torch.device("cpu")

        # Build and pretrain TabNet encoder
        self.encoder = TabNetEncoder(
            input_dim=input_dim, output_dim=output_dim
        ).to(self.device)
        self._decoder = TabNetDecoder(
            output_dim=output_dim, input_dim=input_dim
        ).to(self.device)

        self._pretrain(
            codebook_data,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_size=pretrain_batch_size,
            mask_fraction=mask_fraction,
        )

        # Discard decoder after pretraining
        del self._decoder

        # LinTS on output_dim-dimensional features
        self.d = output_dim
        self.lambda_prior = lambda_prior
        self.nu = nu
        self.B_inv: list[np.ndarray] = [
            np.eye(output_dim) / lambda_prior for _ in range(n_arms)
        ]
        self.f: list[np.ndarray] = [np.zeros(output_dim) for _ in range(n_arms)]

    def _pretrain(
        self,
        data: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
        mask_fraction: float,
    ) -> None:
        """Unsupervised pretraining: masked feature reconstruction."""
        X = torch.from_numpy(data.astype(np.float32)).to(self.device)
        n = X.shape[0]

        params = list(self.encoder.parameters()) + list(self._decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        self.encoder.train()
        self._decoder.train()

        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                batch = X[perm[i : i + batch_size]]
                if batch.shape[0] < 2:
                    continue

                # Random feature masking
                mask = (
                    torch.rand(batch.shape, device=self.device) > mask_fraction
                ).float()
                masked_input = batch * mask

                # Forward
                z = self.encoder(masked_input)
                recon = self._decoder(z)

                # Reconstruction loss only on masked features
                loss = ((recon - batch) ** 2 * (1 - mask)).sum() / (
                    (1 - mask).sum() + 1e-8
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg = total_loss / max(n_batches, 1)
                logger.info(
                    f"TabNet pretrain epoch {epoch + 1}/{epochs}: "
                    f"recon_loss={avg:.4f}"
                )

        self.encoder.eval()

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode raw features to TabNet representation."""
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(x_t)
        return z.cpu().numpy()

    def encode_batch(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Encode a large array in batches."""
        results = []
        for i in range(0, len(X), batch_size):
            results.append(self.encode(X[i : i + batch_size]))
        return np.concatenate(results, axis=0)

    def select_arm(self, context_encoded: np.ndarray) -> int:
        """Select arm via LinTS on TabNet features."""
        x = context_encoded.flatten()
        best_arm = 0
        best_val = -float("inf")

        for a in range(self.n_arms):
            B_inv = self.B_inv[a]
            mu = B_inv @ self.f[a]
            cov = self.nu**2 * B_inv
            cov = (cov + cov.T) / 2
            try:
                L = np.linalg.cholesky(cov)
                theta = mu + L @ self.rng.randn(self.d)
            except np.linalg.LinAlgError:
                cov += 1e-6 * np.eye(self.d)
                try:
                    L = np.linalg.cholesky(cov)
                    theta = mu + L @ self.rng.randn(self.d)
                except np.linalg.LinAlgError:
                    theta = mu + self.rng.randn(self.d) * np.sqrt(
                        np.abs(np.diag(cov))
                    )
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_arm = a

        return best_arm

    def update(self, context_encoded: np.ndarray, arm: int, reward: float) -> None:
        """Update LinTS sufficient statistics (Sherman-Morrison)."""
        x = context_encoded.flatten()
        Bx = self.B_inv[arm] @ x
        denom = 1.0 + float(x @ Bx)
        self.B_inv[arm] = self.B_inv[arm] - np.outer(Bx, Bx) / denom
        self.f[arm] += reward * x
