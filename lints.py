# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Linear Thompson Sampling bandit.

Full-matrix Bayesian linear regression per arm with conjugate
Normal-Inverse-Gamma prior.  At each round, samples weights from the
posterior and selects the arm with the highest predicted reward.
Maintains the precision inverse via Sherman-Morrison for O(d²) updates
instead of O(d³) solves.

Reference: Agrawal & Goyal, "Thompson Sampling for Contextual Bandits
with Linear Payoffs" (ICML 2013).
"""

from __future__ import annotations

import numpy as np


class LinTSBaseline:
    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        lambda_prior: float = 1.0,
        nu: float = 0.1,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.lambda_prior = lambda_prior
        self.nu = nu
        self.rng = rng or np.random.RandomState()

        self.B_inv: list[np.ndarray] = [
            np.eye(input_dim) / lambda_prior for _ in range(n_arms)
        ]
        self.f: list[np.ndarray] = [np.zeros(input_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via Thompson sampling with linear model."""
        x = context.flatten()
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
                    theta = mu + self.rng.randn(self.d) * np.sqrt(np.abs(np.diag(cov)))
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_arm = a

        return best_arm

    def sample_all_arms(self, context: np.ndarray) -> np.ndarray:
        """Sample predicted values for all arms."""
        x = context.flatten()
        vals = np.zeros(self.n_arms)
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
                    theta = mu + self.rng.randn(self.d) * np.sqrt(np.abs(np.diag(cov)))
            vals[a] = float(theta @ x)
        return vals

    def mean_all_arms(self, context: np.ndarray) -> np.ndarray:
        """Get posterior mean predictions for all arms."""
        x = context.flatten()
        vals = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            mu = self.B_inv[a] @ self.f[a]
            vals[a] = float(mu @ x)
        return vals

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update inverse via Sherman-Morrison (O(d²)) and accumulate reward."""
        x = context.flatten()
        Bx = self.B_inv[arm] @ x
        denom = 1.0 + float(x @ Bx)
        self.B_inv[arm] = self.B_inv[arm] - np.outer(Bx, Bx) / denom
        self.f[arm] += reward * x
