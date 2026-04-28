# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Neural-Linear contextual bandit.

Two-phase approach: a neural network learns representations online,
and a LinUCB-style model operates on the last-layer features.
This is the correct way to combine neural representation learning
with principled exploration — the network trains on observed rewards
(no label leakage), and LinUCB provides calibrated uncertainty.

Architecture: input -> hidden -> hidden -> last_layer_features (h).
Decision: LinUCB on h (per-arm ridge regression with UCB).

Reference: Riquelme, Tucker, Snoek, "Deep Bayesian Bandits Showdown"
(ICLR 2018).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


class NeuralLinearBaseline:
    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        hidden_dim: int = 64,
        alpha: float = 1.0,
        lr: float = 1e-3,
        retrain_every: int = 100,
        lambda_reg: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.alpha = alpha
        self.lr = lr
        self.retrain_every = retrain_every
        self.hidden_dim = hidden_dim
        self.step_count = 0

        # Shared feature network: input -> hidden -> hidden -> feature_dim
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.optimizer = torch.optim.Adam(self.feature_net.parameters(), lr=lr)

        # Per-arm linear heads (for training the network end-to-end)
        self.heads: list[nn.Linear] = [nn.Linear(hidden_dim, 1) for _ in range(n_arms)]
        for head in self.heads:
            self.optimizer.add_param_group({"params": head.parameters()})

        # Per-arm LinUCB on last-layer features (for decision-making)
        # +1 for intercept in the linear model
        feat_dim = hidden_dim + 1
        self.A_inv: list[np.ndarray] = [
            np.eye(feat_dim) / lambda_reg for _ in range(n_arms)
        ]
        self.b: list[np.ndarray] = [np.zeros(feat_dim) for _ in range(n_arms)]

        # Replay buffers per arm
        self.replay_x: list[list[np.ndarray]] = [[] for _ in range(n_arms)]
        self.replay_y: list[list[float]] = [[] for _ in range(n_arms)]

    def _get_features(self, x: np.ndarray) -> np.ndarray:
        """Extract last-layer features and append intercept."""
        x_tensor = torch.from_numpy(x.flatten()).float().unsqueeze(0)
        self.feature_net.eval()
        with torch.no_grad():
            h = self.feature_net(x_tensor)
        feat = h.numpy().flatten()
        return np.append(feat, 1.0)  # intercept

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via LinUCB on neural features."""
        feat = self._get_features(context)
        best_arm = 0
        best_val = -float("inf")

        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            ucb = float(theta @ feat) + self.alpha * float(
                np.sqrt(feat @ self.A_inv[a] @ feat)
            )
            if ucb > best_val:
                best_val = ucb
                best_arm = a

        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update linear model and periodically retrain network."""
        x = context.flatten()
        feat = self._get_features(context)

        # Update LinUCB sufficient statistics
        Af = self.A_inv[arm] @ feat
        denom = 1.0 + float(feat @ Af)
        self.A_inv[arm] = self.A_inv[arm] - np.outer(Af, Af) / denom
        self.b[arm] += reward * feat

        # Store in replay buffer
        self.replay_x[arm].append(x.copy())
        self.replay_y[arm].append(reward)

        self.step_count += 1
        if self.step_count % self.retrain_every == 0:
            self._retrain()
            # Reset LinUCB after network changes (features have shifted)
            feat_dim = self.hidden_dim + 1
            self.A_inv = [np.eye(feat_dim) for _ in range(self.n_arms)]
            self.b = [np.zeros(feat_dim) for _ in range(self.n_arms)]
            # Re-populate from replay
            for a in range(self.n_arms):
                for xi, ri in zip(self.replay_x[a], self.replay_y[a]):
                    fi = self._get_features(xi)
                    Af = self.A_inv[a] @ fi
                    denom = 1.0 + float(fi @ Af)
                    self.A_inv[a] = self.A_inv[a] - np.outer(Af, Af) / denom
                    self.b[a] += ri * fi

    def _retrain(self, n_epochs: int = 5) -> None:
        """Retrain feature network on all replay data."""
        total = sum(len(buf) for buf in self.replay_x)
        if total < 20:
            return

        self.feature_net.train()
        for _ in range(n_epochs):
            total_loss = 0.0
            for a in range(self.n_arms):
                if len(self.replay_x[a]) < 5:
                    continue
                X = torch.from_numpy(np.array(self.replay_x[a])).float()
                Y = torch.tensor(self.replay_y[a]).float().unsqueeze(1)
                h = self.feature_net(X)
                pred = self.heads[a](h)
                loss = nn.functional.mse_loss(pred, Y)
                total_loss += loss

            if total_loss > 0:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
