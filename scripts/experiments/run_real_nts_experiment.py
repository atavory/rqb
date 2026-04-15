#!/usr/bin/env python3


"""End-to-end NTS experiment on real tabular datasets.

Validates the full pipeline: real data -> encoder -> RQ trunk codes -> bandit.

For each dataset:
1. Train an MLPEncoder on classification (seeded per encoder_seed)
2. Fit faiss RQ on encoder embeddings to get trunk codes
3. Normalize features with StandardScaler + intercept
4. Build a contextual bandit where trunk codes are contexts and arms
   correspond to actions with context-dependent Bernoulli rewards derived
   from true class distributions
5. Compare HierarchicalTS (trunk codes) vs Random vs LinTS vs RQ-LinTS
   vs ProjRQ vs LinUCB vs RQ-LinUCB vs NeuralUCB vs GLM vs RQ-GLM
   across hyperparameter families optimized via Optuna TPE

Experimental design:
- Outer loop over encoder seeds (captures encoder/RQ variance)
- Inner loop over bandit seeds (captures bandit randomness)
- All RNGs (np.random, torch, per-instance) are seeded for reproducibility

Example:
    python3 scripts/experiments/run_real_nts_experiment.py -- \
        --dataset adult --n-rounds 10000 --n-encoder-seeds 1 --n-bandit-seeds 3
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import logging
from modules.bandits import HierarchicalThompsonSampling
from modules.data import DATASET_REGISTRY, load_dataset
from modules.embeddings import (
    EmbeddingConfig,
    EmbeddingExtractor,
)
from modules.encoders.tab_transformer import MLPEncoder
from modules.features import (
    collect_raw_features,
    FeatureCache,
    normalize_for_rq,
    select_features_supervised,
)
from modules.significance import (
    compute_pairwise_significance,
    log_significance,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LinTSBaseline:
    """Linear Thompson Sampling baseline.

    Bayesian linear regression per arm with conjugate Normal-Inverse-Gamma prior.
    At each round, samples weights from the posterior and selects the arm with
    highest predicted reward.

    Uses a per-instance RNG for reproducibility instead of the global np.random.

    Reference: Agrawal & Goyal, "Thompson Sampling for Contextual Bandits
    with Linear Payoffs" (ICML 2013).
    """

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
        self.nu = nu  # Exploration parameter
        self.rng = rng or np.random.RandomState()

        # Per-arm: maintain B_inv incrementally via Sherman-Morrison (O(d^2))
        # instead of recomputing np.linalg.solve (O(d^3)) every round.
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
            cov = (cov + cov.T) / 2  # symmetrize
            # Use Cholesky instead of numpy's multivariate_normal (which uses
            # SVD internally and can SIGABRT on degenerate matrices).
            try:
                L = np.linalg.cholesky(cov)
                theta = mu + L @ self.rng.randn(self.d)
            except np.linalg.LinAlgError:
                # Not PSD — add diagonal jitter and retry
                cov += 1e-6 * np.eye(self.d)
                try:
                    L = np.linalg.cholesky(cov)
                    theta = mu + L @ self.rng.randn(self.d)
                except np.linalg.LinAlgError:
                    # Last resort — diagonal sampling
                    theta = mu + self.rng.randn(self.d) * np.sqrt(np.abs(np.diag(cov)))
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_arm = a

        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update inverse via Sherman-Morrison (O(d^2)) and accumulate reward."""
        x = context.flatten()
        # Sherman-Morrison: (B + xx^T)^{-1} = B^{-1} - B^{-1}xx^T B^{-1} / (1 + x^T B^{-1} x)
        Bx = self.B_inv[arm] @ x
        denom = 1.0 + float(x @ Bx)
        self.B_inv[arm] = self.B_inv[arm] - np.outer(Bx, Bx) / denom
        self.f[arm] += reward * x


class ProjectedRQLinTS:
    """LinTS on PCA-projected RQ trunk reconstructions.

    With ~2255 unique trunk contexts in 64-D space, PCA to k=8-16
    components should preserve most variance. This achieves O(k^3)
    per decision instead of O(d^3).
    """

    def __init__(
        self,
        trunk_recon: np.ndarray,
        n_components: int,
        n_arms: int,
        lambda_prior: float = 1.0,
        nu: float = 0.1,
        rng: np.random.RandomState | None = None,
    ) -> None:
        n_components = min(n_components, trunk_recon.shape[1])
        self.pca = PCA(n_components=n_components)
        try:
            self.pca.fit(trunk_recon)
        except np.linalg.LinAlgError:
            # SVD can fail to converge on ill-conditioned data; fall back to
            # randomized solver which is more numerically stable.
            self.pca = PCA(n_components=n_components, svd_solver="randomized")
            self.pca.fit(trunk_recon)
        self.explained_variance: float = float(self.pca.explained_variance_ratio_.sum())
        # +1 for intercept
        self.lints = LinTSBaseline(n_components + 1, n_arms, lambda_prior, nu, rng)

    def select_arm(self, features: np.ndarray) -> int:
        """Select arm using PCA-projected trunk reconstruction."""
        projected = self.pca.transform(features)
        projected = _add_intercept(projected)
        return self.lints.select_arm(projected)

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        """Update posterior with PCA-projected features."""
        projected = self.pca.transform(features)
        projected = _add_intercept(projected)
        self.lints.update(projected, arm, reward)


class PCALinTS:
    """LinTS on PCA-projected raw features (dimensionality-reduction baseline).

    Critical ablation: if PCA on raw features matches ProjRQ-k (PCA on RQ
    trunk reconstructions), then RQ adds nothing beyond dimensionality
    reduction. This is the "make-or-break" baseline for the paper.
    """

    def __init__(
        self,
        features: np.ndarray,
        n_components: int,
        n_arms: int,
        lambda_prior: float = 1.0,
        nu: float = 0.1,
        rng: np.random.RandomState | None = None,
    ) -> None:
        n_components = min(n_components, features.shape[1])
        self.pca = PCA(n_components=n_components)
        try:
            self.pca.fit(features)
        except np.linalg.LinAlgError:
            self.pca = PCA(n_components=n_components, svd_solver="randomized")
            self.pca.fit(features)
        self.explained_variance: float = float(self.pca.explained_variance_ratio_.sum())
        # +1 for intercept
        self.lints = LinTSBaseline(n_components + 1, n_arms, lambda_prior, nu, rng)

    def select_arm(self, features: np.ndarray) -> int:
        """Select arm using PCA-projected raw features."""
        projected = self.pca.transform(features)
        projected = _add_intercept(projected)
        return self.lints.select_arm(projected)

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        """Update posterior with PCA-projected features."""
        projected = self.pca.transform(features)
        projected = _add_intercept(projected)
        self.lints.update(projected, arm, reward)


class RandomProjectionLinTS:
    """LinTS on Gaussian random-projected raw features (JL baseline).

    Johnson-Lindenstrauss baseline: random projection preserves pairwise
    distances with high probability. If this matches ProjRQ-k, then even
    the PCA structure is unnecessary — any low-dimensional projection works.
    """

    def __init__(
        self,
        features: np.ndarray,
        n_components: int,
        n_arms: int,
        lambda_prior: float = 1.0,
        nu: float = 0.1,
        rng: np.random.RandomState | None = None,
    ) -> None:
        n_components = min(n_components, features.shape[1])
        seed = rng.randint(2**31) if rng is not None else 42
        self.rp = GaussianRandomProjection(
            n_components=n_components, random_state=seed
        )
        self.rp.fit(features)
        # +1 for intercept
        self.lints = LinTSBaseline(n_components + 1, n_arms, lambda_prior, nu, rng)

    def select_arm(self, features: np.ndarray) -> int:
        """Select arm using random-projected raw features."""
        projected = self.rp.transform(features)
        projected = _add_intercept(projected)
        return self.lints.select_arm(projected)

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        """Update posterior with random-projected features."""
        projected = self.rp.transform(features)
        projected = _add_intercept(projected)
        self.lints.update(projected, arm, reward)


class LinUCBBaseline:
    """LinUCB contextual bandit baseline.

    Per-arm ridge regression with UCB exploration:
        argmax_a (theta_a^T x + alpha * sqrt(x^T A_a^{-1} x))

    Uses Sherman-Morrison O(d^2) incremental inverse updates.

    Reference: Li et al., "A Contextual-Bandit Approach to Personalized News
    Article Recommendation" (WWW 2010).
    """

    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.alpha = alpha

        # Per-arm: A_inv = (X^T X + lambda I)^{-1}, b = X^T y
        self.A_inv: list[np.ndarray] = [
            np.eye(input_dim) / lambda_reg for _ in range(n_arms)
        ]
        self.b: list[np.ndarray] = [np.zeros(input_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via UCB criterion."""
        x = context.flatten()
        best_arm = 0
        best_val = -float("inf")

        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            ucb = float(theta @ x) + self.alpha * float(np.sqrt(x @ self.A_inv[a] @ x))
            if ucb > best_val:
                best_val = ucb
                best_arm = a

        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update inverse via Sherman-Morrison and accumulate reward."""
        x = context.flatten()
        A_inv = self.A_inv[arm]
        # Sherman-Morrison: (A + xx^T)^{-1} = A^{-1} - A^{-1}xx^T A^{-1} / (1 + x^T A^{-1} x)
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom
        self.b[arm] += reward * x


class NeuralUCBBaseline:
    """Neural UCB contextual bandit baseline.

    3-layer MLP reward predictor with gradient-feature UCB:
        UCB = f(x; theta) + gamma * sqrt(g^T Z^{-1} g)
    where g = last-layer features (used as a gradient proxy).

    Uses a replay buffer with periodic retraining.

    Reference: Zhou et al., "Neural Contextual Bandits with UCB-type
    Exploration" (ICML 2020).
    """

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


class GLMBanditBaseline:
    """GLM (Generalized Linear Model) bandit baseline.

    Per-arm logistic regression with confidence ellipsoid UCB:
        UCB = sigmoid(theta_a^T x) + alpha * sqrt(x^T H_a^{-1} x)
    where H is the Fisher information matrix.

    Online gradient updates with clipping for numerical stability.

    Reference: Filippi et al., "Parametric Bandits: The Generalized Linear
    Case" (NeurIPS 2010).
    """

    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        alpha: float = 1.0,
        lr: float = 0.01,
        lambda_reg: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.alpha = alpha
        self.lr = lr

        # Per-arm parameters and Fisher information inverse
        self.theta: list[np.ndarray] = [np.zeros(input_dim) for _ in range(n_arms)]
        self.H_inv: list[np.ndarray] = [
            np.eye(input_dim) / lambda_reg for _ in range(n_arms)
        ]

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return float(1.0 / (1.0 + np.exp(-z)))

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via GLM-UCB criterion."""
        x = context.flatten()
        best_arm = 0
        best_val = -float("inf")

        for a in range(self.n_arms):
            mu = self._sigmoid(float(self.theta[a] @ x))
            ucb = mu + self.alpha * float(np.sqrt(x @ self.H_inv[a] @ x))
            if ucb > best_val:
                best_val = ucb
                best_arm = a

        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Online gradient update and Fisher information update."""
        x = context.flatten()
        mu = self._sigmoid(float(self.theta[arm] @ x))

        # Gradient of log-likelihood for Bernoulli GLM
        grad = (reward - mu) * x
        self.theta[arm] += self.lr * grad

        # Update Fisher information inverse via Sherman-Morrison
        # Fisher info contribution: mu(1-mu) * xx^T
        fisher_weight = mu * (1.0 - mu) + 1e-8  # stability
        w_x = np.sqrt(fisher_weight) * x
        Hw = self.H_inv[arm] @ w_x
        denom = 1.0 + float(w_x @ Hw)
        self.H_inv[arm] = self.H_inv[arm] - np.outer(Hw, Hw) / denom


class NeuralLinearBaseline:
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
        self.heads: list[nn.Linear] = [
            nn.Linear(hidden_dim, 1) for _ in range(n_arms)
        ]
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


class SquareCBBaseline:
    """SquareCB (Inverse Gap Weighting) contextual bandit.

    Elimination-based exploration that does NOT maintain per-arm
    covariance matrices. Uses a reward estimator (online ridge regression)
    and inverse-gap-weighted exploration:
        P(arm a) ∝ 1 / (mu_hat_best - mu_hat_a + gamma)^2

    This achieves near-optimal regret without the O(d^2) per-arm memory
    of LinUCB, making it suitable for high-dimensional contexts.

    Reference: Foster & Rakhlin, "Beyond UCB: Optimal and Efficient
    Contextual Bandits with Regression Oracles" (NeurIPS 2020).
    """

    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        gamma: float = 0.1,
        lambda_reg: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.gamma = gamma
        self.t = 0

        # Shared ridge regression: one model, arm index as one-hot feature
        # Total dim = input_dim + n_arms (context + arm indicator)
        self.total_dim = input_dim + n_arms
        self.A_inv = np.eye(self.total_dim) / lambda_reg
        self.b_vec = np.zeros(self.total_dim)

    def _make_feature(self, context: np.ndarray, arm: int) -> np.ndarray:
        """Concatenate context with one-hot arm indicator."""
        x = context.flatten()
        arm_oh = np.zeros(self.n_arms)
        arm_oh[arm] = 1.0
        return np.concatenate([x, arm_oh])

    def _predict(self, context: np.ndarray, arm: int) -> float:
        """Predict expected reward for (context, arm)."""
        feat = self._make_feature(context, arm)
        theta = self.A_inv @ self.b_vec
        return float(theta @ feat)

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via inverse gap weighting."""
        self.t += 1

        # Get predicted rewards for all arms
        mu_hats = [self._predict(context, a) for a in range(self.n_arms)]
        best_mu = max(mu_hats)

        # Compute inverse gap weights
        # gamma_t decays as 1/sqrt(t) per the paper
        gamma_t = self.gamma / np.sqrt(self.t)
        weights = np.array(
            [1.0 / (best_mu - mu_hats[a] + gamma_t) ** 2 for a in range(self.n_arms)]
        )
        # Normalize to probability
        probs = weights / weights.sum()

        # Sample arm from the distribution
        return int(np.random.choice(self.n_arms, p=probs))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update shared ridge regression model."""
        feat = self._make_feature(context, arm)
        # Sherman-Morrison update
        Af = self.A_inv @ feat
        denom = 1.0 + float(feat @ Af)
        self.A_inv = self.A_inv - np.outer(Af, Af) / denom
        self.b_vec += reward * feat


def _add_intercept(x: np.ndarray) -> np.ndarray:
    """Append a column of 1s to feature matrix for intercept term."""
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate([x, ones], axis=1)


# Per-dataset (nbits, d_cut) from the fast d_cut diagnostic (Appendix D).
# nbits = log2(K) where K is centroids per RQ level: K=8 → nbits=3, K=4 → nbits=2.
DATASET_RQ_CONFIGS: dict[str, tuple[int, int]] = {
    "adult": (3, 3),  # K=8, d=3 → 512 contexts
    "bank_marketing": (3, 3),  # K=8, d=3 → 512 contexts
    "covertype": (3, 5),  # K=8, d=5 → 32,768 contexts
    "higgs": (3, 4),  # K=8, d=4 → 4,096 contexts
    "helena": (2, 3),  # K=4, d=3 → 64 contexts
    "jannis": (3, 2),  # K=8, d=2 → 64 contexts
    "volkert": (3, 2),  # K=8, d=2 → 64 contexts
    "aloi": (2, 3),  # K=4, d=3 → 64 contexts
    "letter": (3, 2),  # K=8, d=2 → 64 contexts
    "dionis": (3, 2),  # K=8, d=2 → 64 contexts
}

# Per-dataset (nbits, d_cut) for raw features trunk codes.
# Populated from dcut_features_sweep results (d_cut diagnostic on raw features).
# Updated 2026-03-04 from MAST dcut diagnostic jobs (raw feature mode).
DATASET_FEATURES_RQ_CONFIGS: dict[str, tuple[int, int]] = {
    "adult": (3, 3),  # K=8, d=3 → 468 ctx, MI=0.131
    "bank_marketing": (3, 3),  # K=8, d=3 → 318 ctx, MI=0.064
    "covertype": (2, 4),  # K=4, d=4 — pending dcut rerun
    "higgs": (2, 4),  # K=4, d=4 — pending dcut rerun
    "helena": (2, 3),  # K=4, d=3 → 61 ctx, MI=0.449
    "jannis": (3, 2),  # K=8, d=2 → 64 ctx, MI=0.192
    "volkert": (2, 4),  # K=4, d=4 → 143 ctx, MI=0.500
    "aloi": (2, 3),  # K=4, d=3 → 46 ctx, MI=1.207
    "letter": (2, 4),  # K=4, d=4 → 222 ctx, MI=0.930
    "dionis": (3, 2),  # K=8, d=2 → 64 ctx, MI=1.821
}

# Per-dataset (nbits, d_cut) for LightGBM leaf-score embeddings.
# Populated from d_cut diagnostic on lgbm embeddings (T=50 trees, 50-dim).
# Best (nbits, d_cut) per dataset chosen by max info_gain across nbits sweep.
DATASET_LGBM_RQ_CONFIGS: dict[str, tuple[int, int]] = {
    "adult": (8, 1),  # K=256, d=1 → 256 contexts, ig=0.120
    "bank_marketing": (8, 1),  # K=256, d=1 → 256 contexts, ig=0.051
    "helena": (6, 1),  # K=64, d=1 → 64 contexts, ig=0.174
    "jannis": (8, 1),  # K=256, d=1 → 256 contexts, ig=0.040
    "volkert": (8, 1),  # K=256, d=1 → 256 contexts, ig=0.267
    "aloi": (6, 1),  # K=64, d=1 → 64 contexts, ig=0.699
    "letter": (6, 1),  # K=64, d=1 → 64 contexts, ig=0.744
    "dionis": (8, 1),  # K=256, d=1 → 256 contexts, ig=1.107
}

# Per-dataset (nbits, d_cut) for Isolation Forest path-depth embeddings.
# Updated 2026-03-04 from MAST dcut diagnostic jobs (iforest feature mode).
# Best (nbits, d_cut) per dataset chosen by max MI across nbits sweep.
DATASET_IFOREST_RQ_CONFIGS: dict[str, tuple[int, int]] = {
    "adult": (8, 1),  # K=256, d=1 → 256 ctx, MI=0.136
    "bank_marketing": (8, 1),  # K=256, d=1 → 256 ctx, MI=0.069
    "helena": (6, 1),  # K=64, d=1 → 64 ctx, MI=0.286
    "jannis": (8, 1),  # K=256, d=1 → 256 ctx, MI=0.069
    "volkert": (8, 1),  # K=256, d=1 → 256 ctx, MI=0.373
    "aloi": (4, 1),  # K=16, d=1 → 16 ctx, MI=0.671
    "letter": (6, 1),  # K=64, d=1 → 64 ctx, MI=0.934
    "dionis": (4, 2),  # K=16, d=2 → 207 ctx, MI=0.764
}


@dataclass
class RealNTSConfig:
    """Configuration for real-data NTS experiments."""

    dataset: str = "adult"
    n_rounds: int = 10000
    n_encoder_seeds: int = 3
    n_bandit_seeds: int = 3
    seed: int = 42

    # Encoder params
    embedding_dim: int = 64
    hidden_dim: int = 128
    encoder_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256

    # RQ params
    n_levels: int = 8
    nbits: int = 6
    d_cut: int = 2  # Shallow trunk for bandit (fewer contexts)

    # Bandit params
    n_arms: int = 5  # Number of bandit arms
    pca_k_values: list[int] = field(default_factory=lambda: [8, 16])
    km_clusters: int | None = None  # K-Means clusters for ablation (None = 2^nbits)

    # Hyperparameter optimization (Optuna TPE)
    n_optuna_trials: int = 20  # Number of Optuna trials
    nu_range: tuple[float, float] = (0.001, 2.0)  # Log-uniform range for nu
    lambda_range: tuple[float, float] = (0.01, 100.0)  # Log-uniform range for lambda
    # Fixed HP mode: if set, skip Optuna and use these values directly
    fixed_nu: float | None = None
    fixed_lambda: float | None = None

    # LinUCB HP ranges
    linucb_alpha_range: tuple[float, float] = (0.01, 5.0)
    linucb_lambda_range: tuple[float, float] = (0.01, 100.0)

    # NeuralUCB HP ranges
    neuralucb_gamma_range: tuple[float, float] = (0.01, 5.0)
    neuralucb_lr_range: tuple[float, float] = (1e-4, 1e-2)

    # GLM HP ranges
    glm_alpha_range: tuple[float, float] = (0.01, 5.0)
    glm_lr_range: tuple[float, float] = (1e-3, 0.1)

    # SquareCB HP ranges
    squarecb_gamma_range: tuple[float, float] = (0.01, 5.0)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature mode: "embeddings" (default), "raw", or "selected"
    feature_mode: str = "embeddings"
    # Cache directory for raw features and feature selection results
    feature_cache_dir: str = "/tmp/feature_cache"
    # Embedding type: "raw" (identity), "lgbm" (LightGBM leaf scores),
    # "iforest" (Isolation Forest path depths), "contrastive" (frozen encoder)
    embedding_type: str = "raw"

    # Optional: restrict to specific HP families (None = run all).
    # Valid values: "TS", "UCB", "NeuralUCB", "GLM", "SquareCB"
    families_to_run: list[str] | None = None


def get_features(
    batch_features: dict[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor | None]:
    """Extract continuous and categorical features from batch."""
    cont_features = batch_features["cont_features"].to(device)
    cat_features = batch_features.get("cat_features")
    if cat_features is not None:
        cat_features = cat_features.to(device)
    return cont_features, cat_features


def pretrain_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,  # pyre-ignore[11]
    n_classes: int,
    config: RealNTSConfig,
    device: torch.device,
) -> nn.Module:
    """Pre-train encoder with classification loss."""
    classifier = nn.Linear(config.embedding_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    encoder.train()
    for epoch in range(config.encoder_epochs):
        total_loss = 0.0
        for batch_features, batch_y in train_loader:
            cont_features, cat_features = get_features(batch_features, device)
            batch_y = batch_y.to(device)

            embeddings = encoder(cat_features=cat_features, cont_features=cont_features)
            logits = classifier(embeddings)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f"  Encoder epoch {epoch + 1}: loss={avg_loss:.4f}")

    return encoder


def collect_embeddings_and_labels(
    encoder: nn.Module,
    loader: DataLoader,  # pyre-ignore[11]
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect all embeddings and labels from a data loader."""
    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for batch_features, batch_y in loader:
            cont_features, cat_features = get_features(batch_features, device)
            emb = encoder(cat_features=cat_features, cont_features=cont_features)
            all_embeddings.append(emb.cpu())
            all_labels.append(batch_y)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def compute_trunk_codes(
    embeddings: Tensor,
    config: RealNTSConfig,
) -> tuple[Tensor, object]:
    """Fit a trunk-only RQ and extract codes from embeddings.

    Uses faiss ResidualQuantizer directly (not FaissMRQModule) since we only
    need trunk codes, not per-class tail codebooks.

    Returns:
        codes: int64 tensor of shape (N, d_cut)
        rq: the trained faiss.ResidualQuantizer (for decoding trunk
            reconstructions)
    """
    import faiss

    embeddings_np = embeddings.detach().numpy().astype(np.float32)
    dim = embeddings_np.shape[1]

    # Create and train a ResidualQuantizer with d_cut levels
    rq = faiss.ResidualQuantizer(dim, config.d_cut, config.nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = False
    rq.train(embeddings_np)

    # Compute codes
    codes_np = rq.compute_codes(embeddings_np)  # (N, d_cut)
    codes = torch.from_numpy(codes_np.astype(np.int64))
    return codes, rq


def build_context_reward_map(
    trunk_codes: Tensor,
    labels: Tensor,
    n_arms: int,
    n_classes: int,
) -> tuple[dict[tuple[int, ...], list[float]], dict[tuple[int, ...], float]]:
    """Build per-context reward probabilities from true class distributions.

    For each unique trunk code context, compute the empirical class distribution.
    Map arms to reward probabilities based on class frequencies:
    - For binary datasets with n_arms > n_classes, spread arms across different
      confidence thresholds derived from class probabilities.
    - For multi-class datasets, arm a gets reward P(class=a | context).

    Returns:
        context_rewards: dict mapping context tuple -> list of reward probs per arm
        context_best: dict mapping context tuple -> best possible reward
    """
    # Group samples by trunk code context
    context_class_counts: dict[tuple[int, ...], list[int]] = defaultdict(
        lambda: [0] * n_classes
    )
    for i in range(trunk_codes.shape[0]):
        key = tuple(trunk_codes[i].tolist())
        label = int(labels[i].item())
        context_class_counts[key][label] += 1

    context_rewards: dict[tuple[int, ...], list[float]] = {}
    context_best: dict[tuple[int, ...], float] = {}

    for key, counts in context_class_counts.items():
        total = sum(counts)
        class_probs = [c / total for c in counts]

        if n_arms <= n_classes:
            # Arms map directly to classes
            arm_rewards = class_probs[:n_arms]
        else:
            # More arms than classes: spread using class probs
            # First n_classes arms map to classes, remaining use blended probs
            arm_rewards = list(class_probs)
            rng_local = np.random.RandomState(hash(key) % (2**31))
            for _ in range(n_arms - n_classes):
                # Blend two random class probs for diversity
                c1, c2 = rng_local.choice(n_classes, 2, replace=True)
                w = rng_local.beta(2, 2)
                arm_rewards.append(w * class_probs[c1] + (1 - w) * class_probs[c2])

        context_rewards[key] = arm_rewards
        context_best[key] = max(arm_rewards)

    return context_rewards, context_best


def _run_bandit_simulation(
    config: RealNTSConfig,
    bandit_seed: int,
    n_arms: int,
    n_codes: int,
    n_samples: int,
    trunk_codes: Tensor,
    embeddings: Tensor,
    embeddings_scaled: np.ndarray,
    trunk_recon_scaled: np.ndarray,
    km_recon_scaled: np.ndarray,
    context_rewards: dict[tuple[int, ...], list[float]],
    context_best: dict[tuple[int, ...], float],
    pca_k_values: list[int],
    nu: float,
    lam: float,
    linucb_alpha: float = 1.0,
    linucb_lambda: float = 1.0,
    neuralucb_gamma: float = 1.0,
    neuralucb_lr: float = 1e-3,
    glm_alpha: float = 1.0,
    glm_lr: float = 0.01,
    squarecb_gamma: float = 0.1,
    methods_to_run: set[str] | None = None,
) -> dict[str, list[float]]:
    """Run a single bandit simulation for one HP config.

    Args:
        methods_to_run: If set, only run these methods. None means all.

    Returns:
        Dictionary mapping method name -> list of cumulative regret at checkpoints.
    """
    rng = np.random.RandomState(bandit_seed)
    np.random.seed(bandit_seed)
    torch.manual_seed(bandit_seed)

    feature_dim = embeddings_scaled.shape[1] + 1  # +1 for intercept

    # Method names
    fixed_methods = [
        "TS",
        "Random",
        "LinTS",
        "RQ-LinTS",
        "KM-LinTS",
        "LinUCB",
        "RQ-LinUCB",
        "KM-LinUCB",
        "NeuralUCB",
        "NeuralLinear",
        "SquareCB",
        "GLM",
        "RQ-GLM",
    ]
    proj_methods = [f"ProjRQ-{k}" for k in pca_k_values]
    pca_methods = [f"PCA-{k}" for k in pca_k_values]
    rp_methods = [f"RP-{k}" for k in pca_k_values]
    all_methods = fixed_methods + proj_methods + pca_methods + rp_methods
    run = methods_to_run if methods_to_run is not None else set(all_methods)

    # Create bandits (only for methods we're running)
    ts_bandit = None
    if "TS" in run:
        ts_bandit = HierarchicalThompsonSampling(
            n_arms=n_arms,
            n_codes=n_codes,
            d_cut=config.d_cut,
        )

    lints = None
    if "LinTS" in run:
        lints = LinTSBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            lambda_prior=lam,
            nu=nu,
            rng=np.random.RandomState(bandit_seed),
        )

    rqlints = None
    if "RQ-LinTS" in run:
        rqlints = LinTSBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            lambda_prior=lam,
            nu=nu,
            rng=np.random.RandomState(bandit_seed + 1),
        )

    kmlints = None
    if "KM-LinTS" in run:
        kmlints = LinTSBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            lambda_prior=lam,
            nu=nu,
            rng=np.random.RandomState(bandit_seed + 3),
        )

    # LinUCB variants
    linucb = None
    if "LinUCB" in run:
        linucb = LinUCBBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            alpha=linucb_alpha,
            lambda_reg=linucb_lambda,
        )

    rq_linucb = None
    if "RQ-LinUCB" in run:
        rq_linucb = LinUCBBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            alpha=linucb_alpha,
            lambda_reg=linucb_lambda,
        )

    km_linucb = None
    if "KM-LinUCB" in run:
        km_linucb = LinUCBBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            alpha=linucb_alpha,
            lambda_reg=linucb_lambda,
        )

    # NeuralUCB (raw embeddings only)
    neural_ucb = None
    if "NeuralUCB" in run:
        neural_ucb = NeuralUCBBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            gamma=neuralucb_gamma,
            lr=neuralucb_lr,
        )

    # GLM variants
    glm = None
    if "GLM" in run:
        glm = GLMBanditBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            alpha=glm_alpha,
            lr=glm_lr,
        )

    rq_glm = None
    if "RQ-GLM" in run:
        rq_glm = GLMBanditBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            alpha=glm_alpha,
            lr=glm_lr,
        )

    # Neural-Linear (same HP family as NeuralUCB)
    neural_linear = None
    if "NeuralLinear" in run:
        neural_linear = NeuralLinearBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            alpha=neuralucb_gamma,  # reuse gamma as alpha for LinUCB head
            lr=neuralucb_lr,
        )

    # SquareCB (uses its own gamma HP)
    squarecb = None
    if "SquareCB" in run:
        squarecb = SquareCBBaseline(
            input_dim=feature_dim,
            n_arms=n_arms,
            gamma=squarecb_gamma,
        )

    # Create ProjRQ bandits for each k
    proj_bandits: dict[int, ProjectedRQLinTS] = {}
    for i, k in enumerate(pca_k_values):
        if f"ProjRQ-{k}" in run:
            proj_bandits[k] = ProjectedRQLinTS(
                trunk_recon=trunk_recon_scaled,
                n_components=k,
                n_arms=n_arms,
                lambda_prior=lam,
                nu=nu,
                rng=np.random.RandomState(bandit_seed + 2 + i),
            )

    # Create PCA-on-raw bandits for each k (dimensionality-reduction baseline)
    pca_bandits: dict[int, PCALinTS] = {}
    for i, k in enumerate(pca_k_values):
        if f"PCA-{k}" in run:
            pca_bandits[k] = PCALinTS(
                features=embeddings_scaled,
                n_components=k,
                n_arms=n_arms,
                lambda_prior=lam,
                nu=nu,
                rng=np.random.RandomState(bandit_seed + 100 + i),
            )

    # Create RP (random projection) bandits for each k (JL baseline)
    rp_bandits: dict[int, RandomProjectionLinTS] = {}
    for i, k in enumerate(pca_k_values):
        if f"RP-{k}" in run:
            rp_bandits[k] = RandomProjectionLinTS(
                features=embeddings_scaled,
                n_components=k,
                n_arms=n_arms,
                lambda_prior=lam,
                nu=nu,
                rng=np.random.RandomState(bandit_seed + 200 + i),
            )

    active_methods = [m for m in all_methods if m in run]
    regret: dict[str, float] = {m: 0.0 for m in active_methods}
    seed_regret: dict[str, list[float]] = {m: [] for m in active_methods}
    checkpoint_interval = max(1, config.n_rounds // 10)

    for round_idx in range(config.n_rounds):
        # Sample a random training example
        idx = rng.randint(n_samples)
        codes = trunk_codes[idx : idx + 1]
        ctx_key = tuple(trunk_codes[idx].tolist())

        rewards_for_ctx = context_rewards[ctx_key]
        best_reward = context_best[ctx_key]

        # HierarchicalTS
        if ts_bandit is not None:
            arm_ts = ts_bandit.select_arm(codes).item()
            reward_ts = float(rng.binomial(1, rewards_for_ctx[arm_ts]))
            ts_bandit.update(codes, torch.tensor([arm_ts]), torch.tensor([reward_ts]))
            regret["TS"] += best_reward - rewards_for_ctx[arm_ts]

        # Random baseline
        if "Random" in run:
            arm_random = rng.randint(n_arms)
            regret["Random"] += best_reward - rewards_for_ctx[arm_random]

        # Precompute context features (only if needed)
        need_emb = (
            lints is not None
            or linucb is not None
            or neural_ucb is not None
            or neural_linear is not None
            or squarecb is not None
            or glm is not None
            or pca_bandits
            or rp_bandits
        )
        need_recon = (
            rqlints is not None
            or rq_linucb is not None
            or rq_glm is not None
            or proj_bandits
        )
        need_km = kmlints is not None or km_linucb is not None

        if need_emb or need_recon or need_km:
            emb_scaled = embeddings_scaled[idx : idx + 1]
            recon_scaled = trunk_recon_scaled[idx : idx + 1]

        if need_emb:
            emb_with_intercept = _add_intercept(emb_scaled)
        if need_recon:
            recon_with_intercept = _add_intercept(recon_scaled)
        if need_km:
            km_scaled = km_recon_scaled[idx : idx + 1]
            km_with_intercept = _add_intercept(km_scaled)

        # LinTS (scaled embeddings + intercept)
        if lints is not None:
            arm_lints = lints.select_arm(emb_with_intercept)
            reward_lints = float(rng.binomial(1, rewards_for_ctx[arm_lints]))
            lints.update(emb_with_intercept, arm_lints, reward_lints)
            regret["LinTS"] += best_reward - rewards_for_ctx[arm_lints]

        # RQ-LinTS (scaled trunk reconstructions + intercept)
        if rqlints is not None:
            arm_rqlints = rqlints.select_arm(recon_with_intercept)
            reward_rqlints = float(rng.binomial(1, rewards_for_ctx[arm_rqlints]))
            rqlints.update(recon_with_intercept, arm_rqlints, reward_rqlints)
            regret["RQ-LinTS"] += best_reward - rewards_for_ctx[arm_rqlints]

        # KM-LinTS (scaled KMeans reconstructions + intercept)
        if kmlints is not None:
            arm_kmlints = kmlints.select_arm(km_with_intercept)
            reward_kmlints = float(rng.binomial(1, rewards_for_ctx[arm_kmlints]))
            kmlints.update(km_with_intercept, arm_kmlints, reward_kmlints)
            regret["KM-LinTS"] += best_reward - rewards_for_ctx[arm_kmlints]

        # LinUCB (raw embeddings)
        if linucb is not None:
            arm_lucb = linucb.select_arm(emb_with_intercept)
            reward_lucb = float(rng.binomial(1, rewards_for_ctx[arm_lucb]))
            linucb.update(emb_with_intercept, arm_lucb, reward_lucb)
            regret["LinUCB"] += best_reward - rewards_for_ctx[arm_lucb]

        # RQ-LinUCB (trunk reconstructions)
        if rq_linucb is not None:
            arm_rqlucb = rq_linucb.select_arm(recon_with_intercept)
            reward_rqlucb = float(rng.binomial(1, rewards_for_ctx[arm_rqlucb]))
            rq_linucb.update(recon_with_intercept, arm_rqlucb, reward_rqlucb)
            regret["RQ-LinUCB"] += best_reward - rewards_for_ctx[arm_rqlucb]

        # KM-LinUCB (KMeans reconstructions)
        if km_linucb is not None:
            arm_kmlucb = km_linucb.select_arm(km_with_intercept)
            reward_kmlucb = float(rng.binomial(1, rewards_for_ctx[arm_kmlucb]))
            km_linucb.update(km_with_intercept, arm_kmlucb, reward_kmlucb)
            regret["KM-LinUCB"] += best_reward - rewards_for_ctx[arm_kmlucb]

        # NeuralUCB (raw embeddings)
        if neural_ucb is not None:
            arm_nucb = neural_ucb.select_arm(emb_with_intercept)
            reward_nucb = float(rng.binomial(1, rewards_for_ctx[arm_nucb]))
            neural_ucb.update(emb_with_intercept, arm_nucb, reward_nucb)
            regret["NeuralUCB"] += best_reward - rewards_for_ctx[arm_nucb]

        # GLM (raw embeddings)
        if glm is not None:
            arm_glm = glm.select_arm(emb_with_intercept)
            reward_glm = float(rng.binomial(1, rewards_for_ctx[arm_glm]))
            glm.update(emb_with_intercept, arm_glm, reward_glm)
            regret["GLM"] += best_reward - rewards_for_ctx[arm_glm]

        # RQ-GLM (trunk reconstructions)
        if rq_glm is not None:
            arm_rqglm = rq_glm.select_arm(recon_with_intercept)
            reward_rqglm = float(rng.binomial(1, rewards_for_ctx[arm_rqglm]))
            rq_glm.update(recon_with_intercept, arm_rqglm, reward_rqglm)
            regret["RQ-GLM"] += best_reward - rewards_for_ctx[arm_rqglm]

        # Neural-Linear (raw embeddings)
        if neural_linear is not None:
            arm_nl = neural_linear.select_arm(emb_with_intercept)
            reward_nl = float(rng.binomial(1, rewards_for_ctx[arm_nl]))
            neural_linear.update(emb_with_intercept, arm_nl, reward_nl)
            regret["NeuralLinear"] += best_reward - rewards_for_ctx[arm_nl]

        # SquareCB (raw embeddings)
        if squarecb is not None:
            arm_sq = squarecb.select_arm(emb_with_intercept)
            reward_sq = float(rng.binomial(1, rewards_for_ctx[arm_sq]))
            squarecb.update(emb_with_intercept, arm_sq, reward_sq)
            regret["SquareCB"] += best_reward - rewards_for_ctx[arm_sq]

        # ProjectedRQLinTS for each k (PCA on scaled trunk recon + intercept)
        for k in pca_k_values:
            if k in proj_bandits:
                proj = proj_bandits[k]
                arm_proj = proj.select_arm(recon_scaled)
                reward_proj = float(rng.binomial(1, rewards_for_ctx[arm_proj]))
                proj.update(recon_scaled, arm_proj, reward_proj)
                regret[f"ProjRQ-{k}"] += best_reward - rewards_for_ctx[arm_proj]

        # PCA-LinTS for each k (PCA on raw features — dim-reduction baseline)
        for k in pca_k_values:
            if k in pca_bandits:
                pca_b = pca_bandits[k]
                arm_pca = pca_b.select_arm(emb_scaled)
                reward_pca = float(rng.binomial(1, rewards_for_ctx[arm_pca]))
                pca_b.update(emb_scaled, arm_pca, reward_pca)
                regret[f"PCA-{k}"] += best_reward - rewards_for_ctx[arm_pca]

        # RP-LinTS for each k (random projection on raw features — JL baseline)
        for k in pca_k_values:
            if k in rp_bandits:
                rp_b = rp_bandits[k]
                arm_rp = rp_b.select_arm(emb_scaled)
                reward_rp = float(rng.binomial(1, rewards_for_ctx[arm_rp]))
                rp_b.update(emb_scaled, arm_rp, reward_rp)
                regret[f"RP-{k}"] += best_reward - rewards_for_ctx[arm_rp]

        if (round_idx + 1) % checkpoint_interval == 0:
            for m in active_methods:
                seed_regret[m].append(regret[m])
            # Progress logging every checkpoint_interval rounds
            pct = 100.0 * (round_idx + 1) / config.n_rounds
            sample_regrets = ", ".join(
                f"{m}={regret[m]:.1f}"
                for m in list(active_methods)[:4]
            )
            print(
                f"    [round {round_idx + 1}/{config.n_rounds} ({pct:.0f}%)] {sample_regrets}",
                flush=True,
            )

    return seed_regret


def run_real_nts(
    config: RealNTSConfig,
    output_dir: str | None = None,
    checkpoint_callback: Callable[..., None] | None = None,
    resume_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """End-to-end: real data -> encoder -> RQ codes -> bandit.

    Outer loop over encoder seeds, inner loop over bandit seeds.
    Uses Optuna TPE sampler to optimize (nu, lambda) per method,
    or runs a single fixed (nu, lambda) if specified.

    Args:
        config: Experiment configuration.
        output_dir: If set, write checkpoint JSONs after each HP family.
        checkpoint_callback: If set, called after each checkpoint with
            ``(family_name, best_final_regrets_so_far, **kwargs)``.
            Additional kwargs include ``state`` (serialized eval cache for
            resumption) and ``partial_result`` (full result dict with
            completed methods, uploaded to the main result path).
        resume_state: If set, pre-populate the eval cache and skip
            already-completed HP families.  Loaded from a previous
            checkpoint's ``state`` dict.
    """
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data (only need to do this once)
    logger.info(f"Loading dataset: {config.dataset}")
    train_dataset, _val_dataset, _test_dataset, metadata = load_dataset(config.dataset)

    n_classes = metadata.n_classes
    logger.info(
        f"Loaded: {metadata.n_samples} samples, "
        f"{metadata.n_continuous} cont, {metadata.n_categorical} cat, "
        f"{n_classes} classes"
    )

    n_arms = max(config.n_arms, n_classes)
    pca_k_values = config.pca_k_values

    # Method names
    fixed_methods = [
        "TS",
        "Random",
        "LinTS",
        "RQ-LinTS",
        "KM-LinTS",
        "LinUCB",
        "RQ-LinUCB",
        "KM-LinUCB",
        "NeuralUCB",
        "NeuralLinear",
        "SquareCB",
        "GLM",
        "RQ-GLM",
    ]
    proj_methods = [f"ProjRQ-{k}" for k in pca_k_values]
    pca_methods = [f"PCA-{k}" for k in pca_k_values]
    rp_methods = [f"RP-{k}" for k in pca_k_values]
    all_methods = fixed_methods + proj_methods + pca_methods + rp_methods

    # HP families: methods sharing the same hyperparameters
    # TS family: nu, lambda (LinTS, RQ-LinTS, KM-LinTS, ProjRQ-k, PCA-k, RP-k)
    ts_family = ["LinTS", "RQ-LinTS", "KM-LinTS"] + proj_methods + pca_methods + rp_methods
    # UCB family: alpha, lambda (LinUCB, RQ-LinUCB, KM-LinUCB)
    ucb_family = ["LinUCB", "RQ-LinUCB", "KM-LinUCB"]
    # NeuralUCB family: gamma, lr (NeuralUCB, NeuralLinear)
    neural_family = ["NeuralUCB", "NeuralLinear"]
    # GLM family: alpha, lr (GLM, RQ-GLM)
    glm_family = ["GLM", "RQ-GLM"]
    # SquareCB family: gamma (SquareCB)
    squarecb_family = ["SquareCB"]
    # HP-free: TS, Random
    hp_free = ["TS", "Random"]

    total_seeds = config.n_encoder_seeds * config.n_bandit_seeds
    logger.info(
        f"Running: {config.n_encoder_seeds} encoder seeds x "
        f"{config.n_bandit_seeds} bandit seeds = {total_seeds} runs"
    )

    # ---------- Precompute encoder/RQ data per encoder seed ----------
    # This is the expensive part — we do it once, then Optuna trials reuse it.
    encoder_data: list[dict[str, object]] = []
    first_n_unique_contexts: int = 0
    first_n_codes: int = 0
    first_pca_ev: dict[int, float] = {}

    # For raw/selected feature modes, precompute features once
    raw_features_scaled: np.ndarray | None = None
    raw_labels: torch.Tensor | None = None
    if config.feature_mode in ("raw", "selected"):
        cache = FeatureCache(config.feature_cache_dir)
        features_np, labels_np = cache.get_or_compute_raw_features(
            config.dataset,
            lambda: collect_raw_features(train_dataset),
        )

        if config.feature_mode == "selected":
            selected_indices, mi_scores = cache.get_or_compute_selection(
                config.dataset,
                "supervised",
                lambda: select_features_supervised(features_np, labels_np),
            )
            features_np = features_np[:, selected_indices]
            logger.info(
                f"  Feature selection: {len(selected_indices)} of "
                f"{mi_scores.shape[0]} features selected (supervised MI)"
            )

        # Normalize before RQ
        raw_features_scaled, _, _ = normalize_for_rq(features_np)
        raw_labels = torch.from_numpy(labels_np)

        # Apply tree-based embedding if requested
        if config.embedding_type in ("lgbm", "iforest"):
            emb_config = EmbeddingConfig(
                embedding_type=config.embedding_type,
            )
            emb_extractor = EmbeddingExtractor(emb_config)
            # lgbm needs labels, iforest doesn't
            raw_features_scaled = emb_extractor.fit_transform(
                features_np,
                y_train=labels_np if config.embedding_type == "lgbm" else None,
            )
            logger.info(
                f"  {config.embedding_type} embeddings: "
                f"{raw_features_scaled.shape[0]} samples, "
                f"{raw_features_scaled.shape[1]} dims"
            )
        else:
            logger.info(
                f"  {config.feature_mode} features: "
                f"{raw_features_scaled.shape[0]} samples, "
                f"{raw_features_scaled.shape[1]} dims"
            )

    # Determine iteration count: raw/selected features are deterministic,
    # so only 1 iteration is needed (no encoder randomness).
    n_feature_iters = (
        1 if config.feature_mode in ("raw", "selected") else config.n_encoder_seeds
    )

    for enc_idx in range(n_feature_iters):
        encoder_seed = config.seed + enc_idx * 10000
        logger.info(
            f"\n  {'Feature' if config.feature_mode != 'embeddings' else 'Encoder'} "
            f"seed {enc_idx + 1}/{n_feature_iters} "
            f"(seed={encoder_seed})"
        )
        print(f"  [{config.dataset}] Seed {enc_idx + 1}/{n_feature_iters}", flush=True)

        # Seed everything
        np.random.seed(encoder_seed)
        torch.manual_seed(encoder_seed)

        if config.feature_mode in ("raw", "selected"):
            # Use precomputed raw/selected features
            assert raw_features_scaled is not None
            assert raw_labels is not None
            embeddings_scaled = raw_features_scaled
            embeddings_scaled_t = torch.from_numpy(embeddings_scaled)
            labels = raw_labels
            n_samples = embeddings_scaled.shape[0]
            embeddings = torch.from_numpy(embeddings_scaled)  # same as scaled for raw
        else:
            # Create data loader with seeded shuffle
            train_loader: DataLoader = DataLoader(  # pyre-ignore[11]
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(encoder_seed),
            )

            # Train encoder
            logger.info("  Pre-training encoder...")
            encoder = MLPEncoder(
                n_categories=metadata.category_sizes,
                n_continuous=metadata.n_continuous,
                dim=config.embedding_dim,
                hidden_dims=[config.hidden_dim, config.hidden_dim],
            ).to(device)
            encoder = pretrain_encoder(encoder, train_loader, n_classes, config, device)

            # Collect embeddings and fit RQ
            logger.info("  Collecting embeddings and fitting RQ...")
            embeddings, labels = collect_embeddings_and_labels(
                encoder, train_loader, device
            )
            n_samples = embeddings.shape[0]
            logger.info(f"  Embeddings shape: {embeddings.shape}")

            # Standardize embeddings BEFORE RQ training so codebook allocation
            # is not biased by feature scale
            embeddings_np = embeddings.detach().numpy().astype(np.float32)
            embeddings_scaled, _, _ = normalize_for_rq(embeddings_np)
            embeddings_scaled_t = torch.from_numpy(embeddings_scaled)

        trunk_codes, trunk_rq = compute_trunk_codes(embeddings_scaled_t, config)
        n_unique_contexts = len(torch.unique(trunk_codes, dim=0))
        logger.info(
            f"  Trunk codes: {trunk_codes.shape}, unique contexts: {n_unique_contexts}"
        )

        # Precompute trunk reconstructions (in scaled space)
        codes_raw = trunk_rq.compute_codes(embeddings_scaled)
        trunk_recon_np: np.ndarray = trunk_rq.decode(codes_raw)

        # Determine n_codes
        n_codes = int(trunk_codes.max().item()) + 1
        n_codes = max(n_codes, 2**config.nbits)

        # embeddings_scaled and trunk_recon are already in standardized space
        trunk_recon_scaled: np.ndarray = trunk_recon_np

        # KMeans reconstruction (ablation: centroid-based denoising)
        km_n_clusters = config.km_clusters or (2**config.nbits)
        km = MiniBatchKMeans(
            n_clusters=min(km_n_clusters, embeddings_scaled.shape[0]),
            random_state=encoder_seed,
            n_init=3,
        )
        km.fit(embeddings_scaled)
        km_recon_scaled: np.ndarray = km.cluster_centers_[km.labels_]

        # Build context -> reward mapping
        logger.info("  Building context-reward mapping...")
        context_rewards, context_best = build_context_reward_map(
            trunk_codes, labels, n_arms, n_classes
        )
        logger.info(f"  Context-reward map: {len(context_rewards)} contexts")

        # Save metadata from first encoder run
        if enc_idx == 0:
            first_n_unique_contexts = n_unique_contexts
            first_n_codes = n_codes
            # Compute PCA explained variance for reporting (on trunk recon)
            for k in pca_k_values:
                k_eff = min(k, trunk_recon_scaled.shape[1])
                pca_temp = PCA(n_components=k_eff)
                try:
                    pca_temp.fit(trunk_recon_scaled)
                except np.linalg.LinAlgError:
                    pca_temp = PCA(n_components=k_eff, svd_solver="randomized")
                    pca_temp.fit(trunk_recon_scaled)
                first_pca_ev[k] = float(pca_temp.explained_variance_ratio_.sum())
            ev_parts = ", ".join(f"k={k}: {first_pca_ev[k]:.3f}" for k in pca_k_values)
            logger.info(f"    PCA on trunk recon EV: {ev_parts}")
            # Compute PCA explained variance on raw features for comparison
            for k in pca_k_values:
                k_eff = min(k, embeddings_scaled.shape[1])
                pca_temp = PCA(n_components=k_eff)
                try:
                    pca_temp.fit(embeddings_scaled)
                except np.linalg.LinAlgError:
                    pca_temp = PCA(n_components=k_eff, svd_solver="randomized")
                    pca_temp.fit(embeddings_scaled)
                ev_raw = float(pca_temp.explained_variance_ratio_.sum())
                logger.info(f"    PCA on raw features k={k}: EV={ev_raw:.3f}")

        encoder_data.append(
            {
                "encoder_seed": encoder_seed,
                "n_samples": n_samples,
                "n_codes": n_codes,
                "trunk_codes": trunk_codes,
                "embeddings": embeddings,
                "embeddings_scaled": embeddings_scaled,
                "trunk_recon_scaled": trunk_recon_scaled,
                "km_recon_scaled": km_recon_scaled,
                "context_rewards": context_rewards,
                "context_best": context_best,
            }
        )

    # ---------- Define the simulation runner ----------
    # HP key type: (nu, lam, linucb_alpha, linucb_lambda, neuralucb_gamma, neuralucb_lr, glm_alpha, glm_lr, squarecb_gamma)
    HPKey = tuple[float, float, float, float, float, float, float, float, float]

    def _evaluate_hp(
        nu: float,
        lam: float,
        linucb_alpha: float = 1.0,
        linucb_lambda: float = 1.0,
        neuralucb_gamma: float = 1.0,
        neuralucb_lr: float = 1e-3,
        glm_alpha: float = 1.0,
        glm_lr: float = 0.01,
        squarecb_gamma: float = 0.1,
        methods_to_run: set[str] | None = None,
    ) -> dict[str, list[list[float]]]:
        """Run all seed pairs for a given HP config. Returns method -> list of regret curves."""
        active = list(methods_to_run) if methods_to_run is not None else all_methods
        hp_regrets: dict[str, list[list[float]]] = {m: [] for m in active}
        n_total_runs = len(encoder_data) * config.n_bandit_seeds
        run_counter = 0

        for ed in encoder_data:
            encoder_seed = ed["encoder_seed"]
            for bs_idx in range(config.n_bandit_seeds):
                bandit_seed = encoder_seed + bs_idx * 1000 + 1
                run_counter += 1
                print(
                    f"    [eval {run_counter}/{n_total_runs}] enc={encoder_seed} bs={bandit_seed} dims={ed['embeddings_scaled'].shape[1]}",
                    flush=True,
                )

                seed_regret = _run_bandit_simulation(
                    config=config,
                    bandit_seed=bandit_seed,
                    n_arms=n_arms,
                    n_codes=ed["n_codes"],
                    n_samples=ed["n_samples"],
                    trunk_codes=ed["trunk_codes"],
                    embeddings=ed["embeddings"],
                    embeddings_scaled=ed["embeddings_scaled"],
                    trunk_recon_scaled=ed["trunk_recon_scaled"],
                    km_recon_scaled=ed["km_recon_scaled"],
                    context_rewards=ed["context_rewards"],
                    context_best=ed["context_best"],
                    pca_k_values=pca_k_values,
                    nu=nu,
                    lam=lam,
                    linucb_alpha=linucb_alpha,
                    linucb_lambda=linucb_lambda,
                    neuralucb_gamma=neuralucb_gamma,
                    neuralucb_lr=neuralucb_lr,
                    glm_alpha=glm_alpha,
                    glm_lr=glm_lr,
                    squarecb_gamma=squarecb_gamma,
                    methods_to_run=methods_to_run,
                )

                for m in active:
                    hp_regrets[m].append(seed_regret[m])

        return hp_regrets

    # ---------- HP optimization via Optuna ----------
    # hp_key -> method -> list of regret curves
    all_hp_results: dict[HPKey, dict[str, list[list[float]]]] = {}
    best_hp_key: dict[str, HPKey] = {}
    best_final: dict[str, float] = {}

    # Default HP values (used when a family's HPs are not being optimized)
    default_nu = 0.1
    default_lam = 1.0
    default_linucb_alpha = 1.0
    default_linucb_lambda = 1.0
    default_neuralucb_gamma = 1.0
    default_neuralucb_lr = 1e-3
    default_glm_alpha = 1.0
    default_glm_lr = 0.01
    default_squarecb_gamma = 0.1

    if config.fixed_nu is not None and config.fixed_lambda is not None:
        # Fixed HP mode — skip Optuna
        nu_fixed = config.fixed_nu
        lam_fixed = config.fixed_lambda
        logger.info(f"\nFixed HP mode: nu={nu_fixed}, lambda={lam_fixed}")
        hp_key: HPKey = (
            nu_fixed,
            lam_fixed,
            default_linucb_alpha,
            default_linucb_lambda,
            default_neuralucb_gamma,
            default_neuralucb_lr,
            default_glm_alpha,
            default_glm_lr,
            default_squarecb_gamma,
        )
        all_hp_results[hp_key] = _evaluate_hp(nu_fixed, lam_fixed)

        for m in all_methods:
            best_hp_key[m] = hp_key
            arr = np.array(all_hp_results[hp_key][m])
            best_final[m] = float(arr[:, -1].mean())

        # Log final regret
        parts = ", ".join(f"{m}={best_final[m]:.1f}" for m in all_methods)
        logger.info(f"  Final regret: {parts}")
        print(f"  [{config.dataset}] Fixed HP final regret: {parts}", flush=True)
    else:
        # Optuna TPE optimization — one study per HP family
        logger.info(
            f"\nOptuna HP optimization: {config.n_optuna_trials} trials per family"
        )

        # Cache to avoid re-running identical HP configs
        _eval_cache: dict[HPKey, dict[str, list[list[float]]]] = {}
        _completed_families: set[str] = set()

        # Resume from saved state if available
        if resume_state:
            for cache_key_str, methods_data in resume_state.get(
                "eval_cache", {}
            ).items():
                parts = [float(x) for x in cache_key_str.split(",")]
                hp_key_r: HPKey = (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                    parts[6],
                    parts[7],
                    parts[8] if len(parts) > 8 else default_squarecb_gamma,
                )
                _eval_cache[hp_key_r] = {
                    m: [list(row) for row in curves]
                    for m, curves in methods_data.items()
                }
            for m, hp_vals in resume_state.get("best_hp_key", {}).items():
                best_hp_key[m] = (
                    hp_vals["nu"],
                    hp_vals["lambda"],
                    hp_vals["linucb_alpha"],
                    hp_vals["linucb_lambda"],
                    hp_vals["neuralucb_gamma"],
                    hp_vals["neuralucb_lr"],
                    hp_vals["glm_alpha"],
                    hp_vals["glm_lr"],
                    hp_vals.get("squarecb_gamma", default_squarecb_gamma),
                )
                best_final[m] = resume_state.get("best_final", {}).get(m, float("inf"))
            _completed_families = set(resume_state.get("completed_families", []))
            logger.info(
                f"Resumed from checkpoint: {len(_eval_cache)} cached configs, "
                f"completed families: {sorted(_completed_families)}"
            )

        def _cached_evaluate(
            nu: float = default_nu,
            lam: float = default_lam,
            linucb_alpha: float = default_linucb_alpha,
            linucb_lambda: float = default_linucb_lambda,
            neuralucb_gamma: float = default_neuralucb_gamma,
            neuralucb_lr: float = default_neuralucb_lr,
            glm_alpha: float = default_glm_alpha,
            glm_lr: float = default_glm_lr,
            squarecb_gamma: float = default_squarecb_gamma,
            methods_to_run: set[str] | None = None,
        ) -> tuple[HPKey, dict[str, list[list[float]]]]:
            # Round to avoid floating-point key mismatches
            key: HPKey = (
                round(nu, 6),
                round(lam, 6),
                round(linucb_alpha, 6),
                round(linucb_lambda, 6),
                round(neuralucb_gamma, 6),
                round(neuralucb_lr, 6),
                round(glm_alpha, 6),
                round(glm_lr, 6),
                round(squarecb_gamma, 6),
            )
            if key not in _eval_cache:
                _eval_cache[key] = _evaluate_hp(
                    nu=nu,
                    lam=lam,
                    linucb_alpha=linucb_alpha,
                    linucb_lambda=linucb_lambda,
                    neuralucb_gamma=neuralucb_gamma,
                    neuralucb_lr=neuralucb_lr,
                    glm_alpha=glm_alpha,
                    glm_lr=glm_lr,
                    squarecb_gamma=squarecb_gamma,
                    methods_to_run=methods_to_run,
                )
            return key, _eval_cache[key]

        _FAMILY_NAMES = {"TS_family", "UCB_family", "NeuralUCB", "SquareCB", "GLM_family"}

        # Family filter: skip families not in config.families_to_run
        _allowed_families: set[str] | None = (
            set(config.families_to_run) if config.families_to_run else None
        )

        def _should_run_family(family_key: str) -> bool:
            """Check if a family should be run (respects families_to_run filter)."""
            if _allowed_families is None:
                return True
            return family_key in _allowed_families

        # Per-family wall-clock timing
        _family_timings: dict[str, float] = {}

        def _inject_cached_trials(
            study: optuna.Study,
            param_names: list[str],
            hp_key_indices: list[int],
            distributions: dict[str, optuna.distributions.LogUniformDistribution],
            representative_method: str,
        ) -> int:
            """Inject cached eval results into an Optuna study for resume.

            This feeds TPE with history so it can make informed suggestions
            instead of starting from scratch after preemption.
            """
            n_injected = 0
            for hp_key, methods_data in _eval_cache.items():
                if representative_method not in methods_data:
                    continue
                params = {
                    name: hp_key[idx]
                    for name, idx in zip(param_names, hp_key_indices)
                }
                arr = np.array(methods_data[representative_method])
                val = float(arr[:, -1].mean())
                frozen = optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    values=[val],
                )
                study.add_trial(frozen)
                n_injected += 1
            if n_injected:
                logger.info(
                    f"[RESUME] Injected {n_injected} cached trials into "
                    f"{representative_method} study"
                )
            return n_injected

        def _serialize_state() -> dict[str, object]:
            """Serialize eval cache + best HP state for resumption."""
            return {
                "eval_cache": {
                    ",".join(str(x) for x in k): {
                        m: (
                            np.array(v).tolist()
                            if isinstance(v, np.ndarray)
                            else [
                                list(row) if isinstance(row, np.ndarray) else row
                                for row in v
                            ]
                        )
                        for m, v in cache_v.items()
                    }
                    for k, cache_v in _eval_cache.items()
                },
                "best_hp_key": {
                    m: {
                        "nu": k[0],
                        "lambda": k[1],
                        "linucb_alpha": k[2],
                        "linucb_lambda": k[3],
                        "neuralucb_gamma": k[4],
                        "neuralucb_lr": k[5],
                        "glm_alpha": k[6],
                        "glm_lr": k[7],
                        "squarecb_gamma": k[8],
                    }
                    for m, k in best_hp_key.items()
                },
                "best_final": {m: float(v) for m, v in best_final.items()},
                "completed_families": sorted(_completed_families),
            }

        def _build_partial_result() -> dict[str, object]:
            """Build a usable result dict from whatever methods are done."""
            checkpoint_interval = max(1, config.n_rounds // 10)
            rounds_list = list(
                range(
                    checkpoint_interval,
                    config.n_rounds + 1,
                    checkpoint_interval,
                )
            )

            # Significance on available methods
            method_fr: dict[str, list[float]] = {}
            for m in all_methods:
                hpk = best_hp_key.get(m)
                if hpk and hpk in _eval_cache and m in _eval_cache[hpk]:
                    arr = np.array(_eval_cache[hpk][m])
                    method_fr[m] = arr[:, -1].tolist()

            sig: dict[str, object] = {}
            if len(method_fr) >= 2:
                try:
                    sig = compute_pairwise_significance(
                        method_fr,
                        ours_methods={"TS", "RQ-LinTS", "RQ-LinUCB", "RQ-GLM"},
                        higher_is_better=False,
                    )
                except Exception:
                    pass

            result: dict[str, object] = {
                "dataset": config.dataset,
                "n_arms": n_arms,
                "n_classes": n_classes,
                "n_unique_contexts": first_n_unique_contexts,
                "n_codes": first_n_codes,
                "d_cut": config.d_cut,
                "n_rounds": config.n_rounds,
                "n_encoder_seeds": config.n_encoder_seeds,
                "n_bandit_seeds": config.n_bandit_seeds,
                "pca_k_values": pca_k_values,
                "rounds": rounds_list,
                "config": asdict(config),
                "best_hp": {
                    m: {
                        "nu": best_hp_key[m][0],
                        "lambda": best_hp_key[m][1],
                        "linucb_alpha": best_hp_key[m][2],
                        "linucb_lambda": best_hp_key[m][3],
                        "neuralucb_gamma": best_hp_key[m][4],
                        "neuralucb_lr": best_hp_key[m][5],
                        "glm_alpha": best_hp_key[m][6],
                        "glm_lr": best_hp_key[m][7],
                    }
                    for m in all_methods
                    if m in best_hp_key
                },
                "significance": sig,
                "completed_families": sorted(_completed_families),
                "partial": len(_completed_families) < 4,
            }

            # Regret curves at each evaluated HP
            for hp_k, hp_regrets in _eval_cache.items():
                hp_label = (
                    f"nu{hp_k[0]}_lam{hp_k[1]}_ua{hp_k[2]}_ul{hp_k[3]}"
                    f"_ng{hp_k[4]}_nl{hp_k[5]}_ga{hp_k[6]}_gl{hp_k[7]}"
                )
                for m in all_methods:
                    if m not in hp_regrets:
                        continue
                    key = m.lower().replace("-", "_")
                    arr = np.array(hp_regrets[m])
                    result[f"regret_{key}_{hp_label}_mean"] = arr.mean(axis=0).tolist()
                    result[f"regret_{key}_{hp_label}_std"] = arr.std(axis=0).tolist()

            return result

        def _write_checkpoint(family_name: str) -> None:
            """Write intermediate results after a trial or HP family."""
            # Grab TS/Random from any cached run that contains them
            if _eval_cache:
                for ck, cr in _eval_cache.items():
                    if hp_free[0] in cr:
                        for m in hp_free:
                            if m not in best_hp_key:
                                best_hp_key[m] = ck
                                arr = np.array(cr[m])
                                best_final[m] = float(arr[:, -1].mean())
                        break

            current_best = {m: best_final[m] for m in all_methods if m in best_final}

            # Serialize state for resumption (every checkpoint)
            state = _serialize_state()

            # Build partial result every checkpoint (including per-trial)
            # so Manifold gets updated results after each Optuna trial.
            partial_result = _build_partial_result()

            # Call external checkpoint callback (e.g. Manifold upload)
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(
                        family_name,
                        current_best,
                        state=state,
                        partial_result=partial_result,
                    )
                except Exception:
                    logger.warning(
                        f"    Checkpoint callback failed for {family_name}: "
                        f"{traceback.format_exc()}"
                    )

            if output_dir is not None:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                suffix = f"_dcut{config.d_cut}" if config.d_cut != 2 else ""
                ckpt_file = (
                    out_path / f"real_nts_{config.dataset}{suffix}_checkpoint.json"
                )
                ckpt: dict[str, object] = {
                    "dataset": config.dataset,
                    "checkpoint_after": family_name,
                    "completed_families": sorted(_completed_families),
                    "best_final_regret": current_best,
                    "n_cached_configs": len(_eval_cache),
                }
                with open(ckpt_file, "w") as f:
                    json.dump(ckpt, f, indent=2, default=float)
                logger.info(f"    Checkpoint written: {ckpt_file}")
            print(
                f"  [{config.dataset}] Checkpoint after {family_name}: "
                + ", ".join(
                    f"{m}={best_final[m]:.1f}" for m in all_methods if m in best_final
                ),
                flush=True,
            )

        # --- TS family: optimize (nu, lambda) ---
        # TS family includes HP-free methods (TS, Random) since they come for free
        if "TS" not in _completed_families and _should_run_family("TS"):
            _ts_start = time.time()
            ts_methods_to_run = set(ts_family + hp_free)
            logger.info(
                "\n  Optimizing TS family (LinTS, RQ-LinTS, KM-LinTS, ProjRQ, PCA, RP)..."
            )

            def _ts_objective(trial: optuna.Trial) -> float:
                nu = trial.suggest_float(
                    "nu", config.nu_range[0], config.nu_range[1], log=True
                )
                lam = trial.suggest_float(
                    "lambda", config.lambda_range[0], config.lambda_range[1], log=True
                )
                try:
                    key, hp_regrets = _cached_evaluate(
                        nu=nu, lam=lam, methods_to_run=ts_methods_to_run
                    )
                except Exception as exc:
                    logger.warning(
                        f"  TS trial {trial.number + 1} FAILED: {exc!r}"
                    )
                    _write_checkpoint(f"TS_trial_{trial.number + 1}")
                    return float("inf")
                # Optimize for the best TS-family method (use RQ-LinTS as representative)
                arr = np.array(hp_regrets["RQ-LinTS"])
                val = float(arr[:, -1].mean())
                print(
                    f"  [{config.dataset}] TS trial {trial.number + 1}/{config.n_optuna_trials}: nu={nu:.4f} lam={lam:.4f} -> RQ-LinTS={val:.1f}",
                    flush=True,
                )
                _write_checkpoint(f"TS_trial_{trial.number + 1}")
                return val

            sampler = optuna.samplers.TPESampler(seed=config.seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            _inject_cached_trials(
                study,
                param_names=["nu", "lambda"],
                hp_key_indices=[0, 1],
                distributions={
                    "nu": optuna.distributions.LogUniformDistribution(
                        low=config.nu_range[0], high=config.nu_range[1],
                    ),
                    "lambda": optuna.distributions.LogUniformDistribution(
                        low=config.lambda_range[0], high=config.lambda_range[1],
                    ),
                },
                representative_method="RQ-LinTS",
            )
            study.optimize(_ts_objective, n_trials=config.n_optuna_trials)

            best_ts_nu = round(study.best_trial.params["nu"], 6)
            best_ts_lam = round(study.best_trial.params["lambda"], 6)
            logger.info(
                f"    TS family: best nu={best_ts_nu:.4f}, lambda={best_ts_lam:.4f}"
            )

            # Pick best HP key for each TS-family method individually
            for method in ts_family:
                best_val = float("inf")
                best_key: HPKey | None = None
                for cached_key, cached_result in _eval_cache.items():
                    if method not in cached_result:
                        continue
                    arr = np.array(cached_result[method])
                    val = float(arr[:, -1].mean())
                    if val < best_val:
                        best_val = val
                        best_key = cached_key
                if best_key is not None:
                    best_hp_key[method] = best_key
                    best_final[method] = best_val
                    logger.info(f"    {method}: regret={best_val:.1f}")

            _completed_families.add("TS")
            _family_timings["TS"] = time.time() - _ts_start
        else:
            logger.info("\n  Skipping TS family (loaded from checkpoint or filtered)")

        _write_checkpoint("TS_family")

        # --- UCB family: optimize (alpha, lambda) ---
        if "UCB" not in _completed_families and _should_run_family("UCB"):
            _ucb_start = time.time()
            ucb_methods_to_run = set(ucb_family)
            logger.info("\n  Optimizing UCB family (LinUCB, RQ-LinUCB, KM-LinUCB)...")

            def _ucb_objective(trial: optuna.Trial) -> float:
                alpha = trial.suggest_float(
                    "alpha",
                    config.linucb_alpha_range[0],
                    config.linucb_alpha_range[1],
                    log=True,
                )
                lam = trial.suggest_float(
                    "lambda",
                    config.linucb_lambda_range[0],
                    config.linucb_lambda_range[1],
                    log=True,
                )
                try:
                    key, hp_regrets = _cached_evaluate(
                        linucb_alpha=alpha,
                        linucb_lambda=lam,
                        methods_to_run=ucb_methods_to_run,
                    )
                except Exception as exc:
                    logger.warning(
                        f"  UCB trial {trial.number + 1} FAILED: {exc!r}"
                    )
                    _write_checkpoint(f"UCB_trial_{trial.number + 1}")
                    return float("inf")
                arr = np.array(hp_regrets["RQ-LinUCB"])
                val = float(arr[:, -1].mean())
                print(
                    f"  [{config.dataset}] UCB trial {trial.number + 1}/{config.n_optuna_trials}: alpha={alpha:.4f} lam={lam:.4f} -> RQ-LinUCB={val:.1f}",
                    flush=True,
                )
                _write_checkpoint(f"UCB_trial_{trial.number + 1}")
                return val

            sampler = optuna.samplers.TPESampler(seed=config.seed + 1)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            _inject_cached_trials(
                study,
                param_names=["alpha", "lambda"],
                hp_key_indices=[2, 3],
                distributions={
                    "alpha": optuna.distributions.LogUniformDistribution(
                        low=config.linucb_alpha_range[0], high=config.linucb_alpha_range[1],
                    ),
                    "lambda": optuna.distributions.LogUniformDistribution(
                        low=config.linucb_lambda_range[0], high=config.linucb_lambda_range[1],
                    ),
                },
                representative_method="RQ-LinUCB",
            )
            study.optimize(_ucb_objective, n_trials=config.n_optuna_trials)

            best_ucb_alpha = round(study.best_trial.params["alpha"], 6)
            best_ucb_lam = round(study.best_trial.params["lambda"], 6)
            logger.info(
                f"    UCB family: best alpha={best_ucb_alpha:.4f}, lambda={best_ucb_lam:.4f}"
            )

            for method in ucb_family:
                best_val = float("inf")
                best_key = None
                for cached_key, cached_result in _eval_cache.items():
                    if method not in cached_result:
                        continue
                    arr = np.array(cached_result[method])
                    val = float(arr[:, -1].mean())
                    if val < best_val:
                        best_val = val
                        best_key = cached_key
                if best_key is not None:
                    best_hp_key[method] = best_key
                    best_final[method] = best_val
                    logger.info(f"    {method}: regret={best_val:.1f}")

            _completed_families.add("UCB")
            _family_timings["UCB"] = time.time() - _ucb_start
        else:
            logger.info("\n  Skipping UCB family (loaded from checkpoint or filtered)")

        _write_checkpoint("UCB_family")

        # --- NeuralUCB family: optimize (gamma, lr) ---
        if "NeuralUCB" not in _completed_families and _should_run_family("NeuralUCB"):
            _neural_start = time.time()
            neural_methods_to_run = set(neural_family)
            logger.info("\n  Optimizing NeuralUCB...")

            def _neural_objective(trial: optuna.Trial) -> float:
                gamma = trial.suggest_float(
                    "gamma",
                    config.neuralucb_gamma_range[0],
                    config.neuralucb_gamma_range[1],
                    log=True,
                )
                lr = trial.suggest_float(
                    "lr",
                    config.neuralucb_lr_range[0],
                    config.neuralucb_lr_range[1],
                    log=True,
                )
                try:
                    key, hp_regrets = _cached_evaluate(
                        neuralucb_gamma=gamma,
                        neuralucb_lr=lr,
                        methods_to_run=neural_methods_to_run,
                    )
                except Exception as exc:
                    logger.warning(
                        f"  Neural trial {trial.number + 1} FAILED: {exc!r}"
                    )
                    _write_checkpoint(f"Neural_trial_{trial.number + 1}")
                    return float("inf")
                arr = np.array(hp_regrets["NeuralUCB"])
                val = float(arr[:, -1].mean())
                print(
                    f"  [{config.dataset}] Neural trial {trial.number + 1}/{config.n_optuna_trials}: gamma={gamma:.4f} lr={lr:.6f} -> NeuralUCB={val:.1f}",
                    flush=True,
                )
                _write_checkpoint(f"Neural_trial_{trial.number + 1}")
                return val

            sampler = optuna.samplers.TPESampler(seed=config.seed + 2)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            _inject_cached_trials(
                study,
                param_names=["gamma", "lr"],
                hp_key_indices=[4, 5],
                distributions={
                    "gamma": optuna.distributions.LogUniformDistribution(
                        low=config.neuralucb_gamma_range[0], high=config.neuralucb_gamma_range[1],
                    ),
                    "lr": optuna.distributions.LogUniformDistribution(
                        low=config.neuralucb_lr_range[0], high=config.neuralucb_lr_range[1],
                    ),
                },
                representative_method="NeuralUCB",
            )
            study.optimize(_neural_objective, n_trials=config.n_optuna_trials)

            best_neural_gamma = round(study.best_trial.params["gamma"], 6)
            best_neural_lr = round(study.best_trial.params["lr"], 6)
            logger.info(
                f"    NeuralUCB: best gamma={best_neural_gamma:.4f}, lr={best_neural_lr:.6f}"
            )

            for method in neural_family:
                best_val = float("inf")
                best_key = None
                for cached_key, cached_result in _eval_cache.items():
                    if method not in cached_result:
                        continue
                    arr = np.array(cached_result[method])
                    val = float(arr[:, -1].mean())
                    if val < best_val:
                        best_val = val
                        best_key = cached_key
                if best_key is not None:
                    best_hp_key[method] = best_key
                    best_final[method] = best_val
                    logger.info(f"    {method}: regret={best_val:.1f}")

            _completed_families.add("NeuralUCB")
            _family_timings["NeuralUCB"] = time.time() - _neural_start
        else:
            logger.info("\n  Skipping NeuralUCB family (loaded from checkpoint or filtered)")

        _write_checkpoint("NeuralUCB")

        # --- GLM family: optimize (alpha, lr) ---
        if "GLM" not in _completed_families and _should_run_family("GLM"):
            _glm_start = time.time()
            glm_methods_to_run = set(glm_family)
            logger.info("\n  Optimizing GLM family (GLM, RQ-GLM)...")

            def _glm_objective(trial: optuna.Trial) -> float:
                alpha = trial.suggest_float(
                    "alpha",
                    config.glm_alpha_range[0],
                    config.glm_alpha_range[1],
                    log=True,
                )
                lr = trial.suggest_float(
                    "lr", config.glm_lr_range[0], config.glm_lr_range[1], log=True
                )
                try:
                    key, hp_regrets = _cached_evaluate(
                        glm_alpha=alpha, glm_lr=lr, methods_to_run=glm_methods_to_run
                    )
                except Exception as exc:
                    logger.warning(
                        f"  GLM trial {trial.number + 1} FAILED: {exc!r}"
                    )
                    _write_checkpoint(f"GLM_trial_{trial.number + 1}")
                    return float("inf")
                arr = np.array(hp_regrets["RQ-GLM"])
                val = float(arr[:, -1].mean())
                print(
                    f"  [{config.dataset}] GLM trial {trial.number + 1}/{config.n_optuna_trials}: alpha={alpha:.4f} lr={lr:.6f} -> RQ-GLM={val:.1f}",
                    flush=True,
                )
                _write_checkpoint(f"GLM_trial_{trial.number + 1}")
                return val

            sampler = optuna.samplers.TPESampler(seed=config.seed + 3)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            _inject_cached_trials(
                study,
                param_names=["alpha", "lr"],
                hp_key_indices=[6, 7],
                distributions={
                    "alpha": optuna.distributions.LogUniformDistribution(
                        low=config.glm_alpha_range[0], high=config.glm_alpha_range[1],
                    ),
                    "lr": optuna.distributions.LogUniformDistribution(
                        low=config.glm_lr_range[0], high=config.glm_lr_range[1],
                    ),
                },
                representative_method="RQ-GLM",
            )
            study.optimize(_glm_objective, n_trials=config.n_optuna_trials)

            best_glm_alpha = round(study.best_trial.params["alpha"], 6)
            best_glm_lr = round(study.best_trial.params["lr"], 6)
            logger.info(
                f"    GLM family: best alpha={best_glm_alpha:.4f}, lr={best_glm_lr:.6f}"
            )

            for method in glm_family:
                best_val = float("inf")
                best_key = None
                for cached_key, cached_result in _eval_cache.items():
                    if method not in cached_result:
                        continue
                    arr = np.array(cached_result[method])
                    val = float(arr[:, -1].mean())
                    if val < best_val:
                        best_val = val
                        best_key = cached_key
                if best_key is not None:
                    best_hp_key[method] = best_key
                    best_final[method] = best_val
                    logger.info(f"    {method}: regret={best_val:.1f}")

            _completed_families.add("GLM")
            _family_timings["GLM"] = time.time() - _glm_start
        else:
            logger.info("\n  Skipping GLM family (loaded from checkpoint or filtered)")

        _write_checkpoint("GLM_family")

        # --- SquareCB family: optimize gamma ---
        if "SquareCB" not in _completed_families and _should_run_family("SquareCB"):
            _sqcb_start = time.time()
            sqcb_methods_to_run = set(squarecb_family)
            logger.info("\n  Optimizing SquareCB family...")

            def _squarecb_objective(trial: optuna.Trial) -> float:
                gamma = trial.suggest_float(
                    "gamma",
                    config.squarecb_gamma_range[0],
                    config.squarecb_gamma_range[1],
                    log=True,
                )
                key, hp_regrets = _cached_evaluate(
                    squarecb_gamma=gamma, methods_to_run=sqcb_methods_to_run
                )
                arr = np.array(hp_regrets["SquareCB"])
                val = float(arr[:, -1].mean())
                print(
                    f"  [{config.dataset}] SquareCB trial {trial.number + 1}/{config.n_optuna_trials}: gamma={gamma:.4f} -> SquareCB={val:.1f}",
                    flush=True,
                )
                _write_checkpoint(f"SquareCB_trial_{trial.number + 1}")
                return val

            sampler = optuna.samplers.TPESampler(seed=config.seed + 4)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            _inject_cached_trials(
                study,
                param_names=["gamma"],
                hp_key_indices=[8],
                distributions={
                    "gamma": optuna.distributions.LogUniformDistribution(
                        low=config.squarecb_gamma_range[0], high=config.squarecb_gamma_range[1],
                    ),
                },
                representative_method="SquareCB",
            )
            study.optimize(_squarecb_objective, n_trials=config.n_optuna_trials)

            best_sqcb_gamma = round(study.best_trial.params["gamma"], 6)
            logger.info(f"    SquareCB family: best gamma={best_sqcb_gamma:.4f}")

            for method in squarecb_family:
                best_val = float("inf")
                best_key = None
                for cached_key, cached_result in _eval_cache.items():
                    if method not in cached_result:
                        continue
                    arr = np.array(cached_result[method])
                    val = float(arr[:, -1].mean())
                    if val < best_val:
                        best_val = val
                        best_key = cached_key
                if best_key is not None:
                    best_hp_key[method] = best_key
                    best_final[method] = best_val
                    logger.info(f"    {method}: regret={best_val:.1f}")

            _completed_families.add("SquareCB")
            _family_timings["SquareCB"] = time.time() - _sqcb_start
        else:
            logger.info("\n  Skipping SquareCB family (loaded from checkpoint or filtered)")

        _write_checkpoint("SquareCB")

        # TS and Random don't use HP — grab from any cache entry that has them
        for cached_key, cached_result in _eval_cache.items():
            if hp_free[0] in cached_result:
                for m in hp_free:
                    best_hp_key[m] = cached_key
                    arr = np.array(cached_result[m])
                    best_final[m] = float(arr[:, -1].mean())
                break

        _write_checkpoint("all_families")

        # Store all evaluated HP configs
        all_hp_results = dict(_eval_cache)

    # ---------- Report results ----------
    checkpoint_interval = max(1, config.n_rounds // 10)
    rounds = list(range(checkpoint_interval, config.n_rounds + 1, checkpoint_interval))

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS: {config.dataset} (n_arms={n_arms}, d_cut={config.d_cut})")
    logger.info("=" * 80)

    # Best HP per method
    logger.info("\nBest hyperparameters per method:")
    for m in all_methods:
        if m in best_hp_key:
            hp_k = best_hp_key[m]
            logger.info(
                f"  {m:>12s}: final_regret={best_final.get(m, float('nan')):.1f}  "
                f"[nu={hp_k[0]:.4f}, λ={hp_k[1]:.4f}, "
                f"ucb_α={hp_k[2]:.4f}, ucb_λ={hp_k[3]:.4f}, "
                f"γ={hp_k[4]:.4f}, n_lr={hp_k[5]:.6f}, "
                f"glm_α={hp_k[6]:.4f}, glm_lr={hp_k[7]:.6f}, "
                f"sq_γ={hp_k[8]:.4f}]"
            )

    # Detailed regret curves at each method's best HP
    logger.info("\nRegret curves at each method's best HP:")
    header = f"{'Round':>7s}" + "".join(f"  {m:>12s}" for m in all_methods)
    logger.info(header)
    logger.info("-" * len(header))

    for i, r in enumerate(rounds):
        row = f"{r:7d}"
        for m in all_methods:
            hp_key = best_hp_key.get(m)
            if hp_key and hp_key in all_hp_results and m in all_hp_results[hp_key]:
                arr = np.array(all_hp_results[hp_key][m])
                row += f"  {arr[:, i].mean():12.1f}"
            else:
                row += f"  {'N/A':>12s}"
        logger.info(row)

    logger.info("-" * len(header))
    logger.info(f"Unique contexts: {first_n_unique_contexts}")
    logger.info(f"n_codes: {first_n_codes}, d_cut: {config.d_cut}")

    # Final comparison vs LinTS
    final_lints = best_final.get("LinTS", 0)
    if final_lints > 0:
        logger.info("\nFinal regret vs LinTS (each method at its own best HP):")
        for m in all_methods:
            final_m = best_final.get(m, float("nan"))
            hp_key = best_hp_key.get(m)
            std_m = 0.0
            if hp_key and hp_key in all_hp_results and m in all_hp_results[hp_key]:
                arr = np.array(all_hp_results[hp_key][m])
                std_m = float(arr[:, -1].std())
            ratio = final_m / final_lints if final_lints > 0 else float("nan")
            logger.info(
                f"  {m:>12s}: {final_m:8.1f} +/- {std_m:6.1f}  ({ratio:.2f}x LinTS)"
            )

    # ---------- Significance testing ----------
    # Collect final regret per seed for each method at its best HP
    method_final_regrets: dict[str, list[float]] = {}
    for m in all_methods:
        hp_key = best_hp_key.get(m)
        if hp_key and hp_key in all_hp_results and m in all_hp_results[hp_key]:
            arr = np.array(all_hp_results[hp_key][m])
            method_final_regrets[m] = arr[:, -1].tolist()

    sig: dict[str, object] = {}
    if method_final_regrets:
        sig = compute_pairwise_significance(
            method_final_regrets,
            ours_methods={"TS", "RQ-LinTS", "RQ-LinUCB", "RQ-GLM"},
            higher_is_better=False,
        )
        if sig:
            logger.info("\n" + "=" * 80)
            logger.info("SIGNIFICANCE TESTS (best-ours vs best-baseline)")
            logger.info("=" * 80)
            log_significance(sig, logger, metric_name="final regret")

    # Build result dict
    result: dict[str, object] = {
        "dataset": config.dataset,
        "n_arms": n_arms,
        "n_classes": n_classes,
        "n_unique_contexts": first_n_unique_contexts,
        "n_codes": first_n_codes,
        "d_cut": config.d_cut,
        "n_rounds": config.n_rounds,
        "n_encoder_seeds": config.n_encoder_seeds,
        "n_bandit_seeds": config.n_bandit_seeds,
        "pca_k_values": pca_k_values,
        "rounds": rounds,
        "config": asdict(config),
        "best_hp": {
            m: {
                "nu": best_hp_key[m][0],
                "lambda": best_hp_key[m][1],
                "linucb_alpha": best_hp_key[m][2],
                "linucb_lambda": best_hp_key[m][3],
                "neuralucb_gamma": best_hp_key[m][4],
                "neuralucb_lr": best_hp_key[m][5],
                "glm_alpha": best_hp_key[m][6],
                "glm_lr": best_hp_key[m][7],
                "squarecb_gamma": best_hp_key[m][8],
            }
            for m in all_methods
            if m in best_hp_key
        },
        "significance": sig,
        "family_timings_seconds": _family_timings,
    }

    # Store per-HP results
    for hp_k, hp_regrets in all_hp_results.items():
        hp_label = f"nu{hp_k[0]}_lam{hp_k[1]}_ua{hp_k[2]}_ul{hp_k[3]}_ng{hp_k[4]}_nl{hp_k[5]}_ga{hp_k[6]}_gl{hp_k[7]}_sg{hp_k[8]}"
        for m in all_methods:
            if m not in hp_regrets:
                continue
            key = m.lower().replace("-", "_")
            arr = np.array(hp_regrets[m])
            result[f"regret_{key}_{hp_label}_mean"] = arr.mean(axis=0).tolist()
            result[f"regret_{key}_{hp_label}_std"] = arr.std(axis=0).tolist()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-data NTS experiment: real dataset -> encoder -> RQ -> bandit"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=list(DATASET_REGISTRY.keys() - {"synthetic"}),
        help="Dataset to use",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=10000, help="Number of bandit rounds"
    )
    parser.add_argument(
        "--n-encoder-seeds",
        type=int,
        default=3,
        help="Number of encoder/RQ random seeds (outer loop)",
    )
    parser.add_argument(
        "--n-bandit-seeds",
        type=int,
        default=3,
        help="Number of bandit random seeds (inner loop)",
    )
    parser.add_argument(
        "--n-arms",
        type=int,
        default=5,
        help="Number of bandit arms (at least n_classes)",
    )
    parser.add_argument(
        "--d-cut", type=int, default=2, help="Trunk depth for context codes"
    )
    parser.add_argument("--nbits", type=int, default=6, help="Bits per RQ level")
    parser.add_argument(
        "--auto-dcut",
        action="store_true",
        help="Auto-set nbits and d_cut from per-dataset diagnostic configs",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=64, help="Encoder embedding dimension"
    )
    parser.add_argument(
        "--encoder-epochs", type=int, default=20, help="Encoder pre-training epochs"
    )
    parser.add_argument(
        "--pca-k",
        type=int,
        nargs="+",
        default=[8, 16],
        help="PCA projection dimensions for ProjRQ bandits",
    )
    parser.add_argument(
        "--km-clusters",
        type=int,
        default=None,
        help="Number of K-Means clusters for KM-LinTS ablation (default: 2^nbits)",
    )
    parser.add_argument(
        "--n-optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna TPE trials per HP family for optimization",
    )
    parser.add_argument(
        "--fixed-nu",
        type=float,
        default=None,
        help="Fixed nu (skip Optuna if both --fixed-nu and --fixed-lambda are set)",
    )
    parser.add_argument(
        "--fixed-lambda",
        type=float,
        default=None,
        help="Fixed lambda (skip Optuna if both --fixed-nu and --fixed-lambda are set)",
    )
    parser.add_argument(
        "--nu-range",
        type=float,
        nargs=2,
        default=[0.001, 2.0],
        help="Log-uniform range for nu (min max)",
    )
    parser.add_argument(
        "--lambda-range",
        type=float,
        nargs=2,
        default=[0.01, 100.0],
        help="Log-uniform range for lambda (min max)",
    )
    parser.add_argument(
        "--linucb-alpha-range",
        type=float,
        nargs=2,
        default=[0.01, 5.0],
        help="Log-uniform range for LinUCB alpha (min max)",
    )
    parser.add_argument(
        "--linucb-lambda-range",
        type=float,
        nargs=2,
        default=[0.01, 100.0],
        help="Log-uniform range for LinUCB lambda (min max)",
    )
    parser.add_argument(
        "--neuralucb-gamma-range",
        type=float,
        nargs=2,
        default=[0.01, 5.0],
        help="Log-uniform range for NeuralUCB gamma (min max)",
    )
    parser.add_argument(
        "--neuralucb-lr-range",
        type=float,
        nargs=2,
        default=[1e-4, 1e-2],
        help="Log-uniform range for NeuralUCB learning rate (min max)",
    )
    parser.add_argument(
        "--glm-alpha-range",
        type=float,
        nargs=2,
        default=[0.01, 5.0],
        help="Log-uniform range for GLM alpha (min max)",
    )
    parser.add_argument(
        "--glm-lr-range",
        type=float,
        nargs=2,
        default=[1e-3, 0.1],
        help="Log-uniform range for GLM learning rate (min max)",
    )
    parser.add_argument(
        "--squarecb-gamma-range",
        type=float,
        nargs=2,
        default=[0.01, 5.0],
        help="Log-uniform range for SquareCB gamma (min max)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/real_nts_results",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for encoder training (default: cuda if available, else cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="embeddings",
        choices=["embeddings", "raw", "selected"],
        help="Input space for RQ: encoder embeddings, raw features, or selected features",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="results/feature_cache",
        help="Directory for caching raw features and feature selection results",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="raw",
        choices=["raw", "lgbm", "iforest", "contrastive"],
        help="Embedding transformation before RQ: raw (identity), lgbm (leaf scores), iforest (path depths)",
    )
    args = parser.parse_args()

    # Auto-set nbits and d_cut from per-dataset diagnostic configs
    nbits = args.nbits
    d_cut = args.d_cut
    if args.auto_dcut:
        # Dispatch on embedding_type first, then feature_mode
        if args.embedding_type == "lgbm":
            rq_configs = DATASET_LGBM_RQ_CONFIGS
            config_label = "lgbm"
        elif args.embedding_type == "iforest":
            rq_configs = DATASET_IFOREST_RQ_CONFIGS
            config_label = "iforest"
        elif args.feature_mode in ("raw", "selected"):
            rq_configs = DATASET_FEATURES_RQ_CONFIGS
            config_label = "features"
        else:
            rq_configs = DATASET_RQ_CONFIGS
            config_label = "embeddings"
        if args.dataset in rq_configs:
            nbits, d_cut = rq_configs[args.dataset]
            logger.info(
                f"Auto-dcut ({config_label}): {args.dataset} -> nbits={nbits}, d_cut={d_cut}"
            )
        else:
            logger.info(
                f"Auto-dcut ({config_label}): no config for {args.dataset}, using defaults "
                f"nbits={nbits}, d_cut={d_cut}"
            )

    config = RealNTSConfig(
        dataset=args.dataset,
        n_rounds=args.n_rounds,
        n_encoder_seeds=args.n_encoder_seeds,
        n_bandit_seeds=args.n_bandit_seeds,
        n_arms=args.n_arms,
        d_cut=d_cut,
        nbits=nbits,
        embedding_dim=args.embedding_dim,
        encoder_epochs=args.encoder_epochs,
        pca_k_values=args.pca_k,
        km_clusters=args.km_clusters,
        n_optuna_trials=args.n_optuna_trials,
        fixed_nu=args.fixed_nu,
        fixed_lambda=args.fixed_lambda,
        nu_range=tuple(args.nu_range),
        lambda_range=tuple(args.lambda_range),
        linucb_alpha_range=tuple(args.linucb_alpha_range),
        linucb_lambda_range=tuple(args.linucb_lambda_range),
        neuralucb_gamma_range=tuple(args.neuralucb_gamma_range),
        neuralucb_lr_range=tuple(args.neuralucb_lr_range),
        glm_alpha_range=tuple(args.glm_alpha_range),
        glm_lr_range=tuple(args.glm_lr_range),
        squarecb_gamma_range=tuple(args.squarecb_gamma_range),
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
        feature_mode=args.feature_mode,
        feature_cache_dir=args.feature_cache_dir,
        embedding_type=args.embedding_type,
    )

    logger.info("=" * 60)
    logger.info("REAL-DATA NTS EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset}, feature_mode: {config.feature_mode}")
    logger.info(
        f"n_rounds: {config.n_rounds}, "
        f"n_encoder_seeds: {config.n_encoder_seeds}, "
        f"n_bandit_seeds: {config.n_bandit_seeds}"
    )
    logger.info(f"d_cut: {config.d_cut}, nbits: {config.nbits}")
    logger.info(f"n_arms: {config.n_arms}, pca_k: {config.pca_k_values}")
    if config.fixed_nu is not None and config.fixed_lambda is not None:
        logger.info(f"Fixed HP: nu={config.fixed_nu}, lambda={config.fixed_lambda}")
    else:
        logger.info(
            f"Optuna TPE: {config.n_optuna_trials} trials per family, "
            f"nu in [{config.nu_range[0]}, {config.nu_range[1]}], "
            f"lambda in [{config.lambda_range[0]}, {config.lambda_range[1]}]"
        )

    start_time = time.time()
    results = run_real_nts(config, output_dir=args.output_dir)
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_dcut{config.d_cut}" if config.d_cut != 2 else ""
    output_file = output_dir / f"real_nts_{config.dataset}{suffix}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
