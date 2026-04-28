# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Neural UCB contextual bandit baseline.

3-layer MLP reward predictor with gradient-feature UCB:
    UCB = f(x; theta) + gamma * sqrt(g^T Z^{-1} g)
where g = last-layer features (used as a gradient proxy).

Uses a replay buffer with periodic retraining.

Reference: Zhou et al., "Neural Contextual Bandits with UCB-type
Exploration" (ICML 2020).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class NeuralUCBBaseline:
    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        hidden_dim: int = 64,
        gamma: float = 1.0,
        lr: float = 1e-3,
        retrain_every: int = 100,
        lambda_reg: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.gamma = gamma
        self.lr = lr
        self.retrain_every = retrain_every
        self.step_count = 0

        # Per-arm MLP: input -> hidden -> hidden -> 1
        self.networks: list[nn.Sequential] = []
        self.optimizers: list[torch.optim.Adam] = []
        for _ in range(n_arms):
            net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.networks.append(net)
            self.optimizers.append(torch.optim.Adam(net.parameters(), lr=lr))

        # Per-arm Z matrix for UCB (on last-layer features)
        self.last_dim = hidden_dim
        self.Z_inv: list[np.ndarray] = [
            np.eye(hidden_dim) / lambda_reg for _ in range(n_arms)
        ]

        # Replay buffers per arm
        self.replay_x: list[list[np.ndarray]] = [[] for _ in range(n_arms)]
        self.replay_y: list[list[float]] = [[] for _ in range(n_arms)]

    def _get_last_layer_features(
        self, net: nn.Sequential, x_tensor: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning (prediction, last-layer features)."""
        h = x_tensor
        # Pass through all but last layer to get features
        for layer in list(net.children())[:-1]:
            h = layer(h)
        features = h.detach()
        pred = net[-1](h)
        return pred, features

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via Neural UCB criterion."""
        x = context.flatten()
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        best_arm = 0
        best_val = -float("inf")

        for a in range(self.n_arms):
            self.networks[a].eval()
            with torch.no_grad():
                pred, feat = self._get_last_layer_features(self.networks[a], x_tensor)
            g = feat.numpy().flatten()
            ucb = float(pred.item()) + self.gamma * float(
                np.sqrt(g @ self.Z_inv[a] @ g)
            )
            if ucb > best_val:
                best_val = ucb
                best_arm = a

        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Store experience and periodically retrain."""
        x = context.flatten()
        self.replay_x[arm].append(x.copy())
        self.replay_y[arm].append(reward)

        # Update Z_inv with Sherman-Morrison using last-layer features
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        self.networks[arm].eval()
        with torch.no_grad():
            _, feat = self._get_last_layer_features(self.networks[arm], x_tensor)
        g = feat.numpy().flatten()
        Zg = self.Z_inv[arm] @ g
        denom = 1.0 + float(g @ Zg)
        self.Z_inv[arm] = self.Z_inv[arm] - np.outer(Zg, Zg) / denom

        self.step_count += 1
        if self.step_count % self.retrain_every == 0:
            self._retrain(arm)

    def _retrain(self, arm: int, n_epochs: int = 5) -> None:
        """Retrain network on replay buffer."""
        if len(self.replay_x[arm]) < 10:
            return
        net = self.networks[arm]
        opt = self.optimizers[arm]
        X = torch.from_numpy(np.array(self.replay_x[arm])).float()
        Y = torch.tensor(self.replay_y[arm]).float().unsqueeze(1)

        net.train()
        for _ in range(n_epochs):
            pred = net(X)
            loss = nn.functional.mse_loss(pred, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
