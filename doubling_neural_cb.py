# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Doubling Neural Contextual Bandit.

Epoch-doubling schedule: starts with ``init_phase`` rounds of random
exploration, then trains a 2-layer MLP reward predictor.  Each subsequent
phase doubles in length (500 → 1000 → 2000 → …).  At each phase boundary
the network is retrained on the full replay buffer.

Within a phase the agent plays greedily according to the current model
(or randomly if no model has been trained yet).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger: logging.Logger = logging.getLogger(__name__)


class DoublingNeuralCB:
    """MLP contextual bandit with doubling-epoch retraining."""

    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        hidden_dim: int = 64,
        init_phase: int = 500,
        lr: float = 1e-3,
        train_epochs: int = 10,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.n_arms = n_arms
        self.hidden_dim = hidden_dim
        self.init_phase = init_phase
        self.lr = lr
        self.train_epochs = train_epochs
        self.rng = rng or np.random.RandomState()

        self.step_count = 0
        self.phase_len = init_phase
        self.next_train_at = init_phase  # train after first phase

        # Per-arm MLP: input -> hidden -> hidden -> 1
        self.networks: list[nn.Sequential] = []
        for _ in range(n_arms):
            net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.networks.append(net)

        self.trained = False

        # Replay buffers per arm
        self.replay_x: list[list[np.ndarray]] = [[] for _ in range(n_arms)]
        self.replay_y: list[list[float]] = [[] for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        if not self.trained:
            return int(self.rng.randint(self.n_arms))

        x = context.flatten()
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        best_arm = 0
        best_val = -float("inf")
        for a in range(self.n_arms):
            self.networks[a].eval()
            with torch.no_grad():
                val = float(self.networks[a](x_tensor).item())
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        self.replay_x[arm].append(context.flatten().copy())
        self.replay_y[arm].append(reward)
        self.step_count += 1

        if self.step_count >= self.next_train_at:
            self._retrain_all()
            self.trained = True
            self.phase_len *= 2
            self.next_train_at = self.step_count + self.phase_len

    def _retrain_all(self) -> None:
        for a in range(self.n_arms):
            if len(self.replay_x[a]) < 5:
                continue
            net = self.networks[a]
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            X = torch.from_numpy(np.array(self.replay_x[a])).float()
            Y = torch.tensor(self.replay_y[a]).float().unsqueeze(1)

            net.train()
            for _ in range(self.train_epochs):
                pred = net(X)
                loss = nn.functional.mse_loss(pred, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
