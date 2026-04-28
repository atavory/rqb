# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Diagonal (Adagrad-style) Linear Thompson Sampling bandit.

Per-arm diagonal-precision Bayesian linear model.  Exploration via
diagonal posterior sampling.  Update via rank-1 precision accumulation
(equivalent to Adagrad-preconditioned SGD on squared loss).  O(d) per
step, compared to O(d²) for full-matrix LinTS.
"""

from __future__ import annotations

import numpy as np


class SGDLinTS:
    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        nu: float = 0.1,
        lam: float = 1.0,
        alpha: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_arms = n_arms
        self.input_dim = input_dim
        self.nu = nu
        self.alpha = alpha
        self.prec = [np.full(input_dim, lam, dtype=np.float64) for _ in range(n_arms)]
        self.f = [np.zeros(input_dim, dtype=np.float64) for _ in range(n_arms)]
        self.rng = np.random.RandomState(seed)

    def select_arm(self, x: np.ndarray) -> int:
        """Select arm via diagonal Thompson Sampling."""
        x = x.flatten()
        best_arm = 0
        best_val = -float("inf")
        for a in range(self.n_arms):
            mu = self.f[a] / self.prec[a]
            sigma = self.nu / np.sqrt(self.prec[a])
            theta_sample = mu + sigma * self.rng.randn(self.input_dim)
            score = float(theta_sample @ x)
            if score > best_val:
                best_val = score
                best_arm = a
        return best_arm

    def predict(self, x: np.ndarray, arm: int) -> float:
        """Posterior mean prediction."""
        mu = self.f[arm] / self.prec[arm]
        return float(mu @ x.flatten())

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        """Bayesian diagonal update — O(d)."""
        x = x.flatten()
        self.prec[arm] += x * x
        self.f[arm] += reward * x
