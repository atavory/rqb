# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""Normal-Inverse-Gamma sufficient statistics for Bayesian bandits.

Provides a conjugate prior for the Normal distribution with unknown mean
and variance. Each arm in a contextual bandit maintains a NIG prior that
is updated with observed rewards, enabling Thompson Sampling via
posterior sampling.

NIG(μ₀, λ, α, β) parametrisation:
    μ₀  — prior mean
    λ   — pseudo-count (precision scale)
    α   — shape  (≥ 1 for proper prior)
    β   — rate   (> 0)

Posterior update given observation x:
    λ' = λ + 1
    μ₀' = (λ·μ₀ + x) / λ'
    α' = α + 0.5
    β' = β + 0.5·λ·(x - μ₀)² / λ'
"""

from __future__ import annotations

import torch
from torch import Tensor


class NIGStats:
    """Normal-Inverse-Gamma sufficient statistics for a set of arms.

    Stores per-arm NIG parameters as tensors for vectorized posterior
    updates and Thompson Sampling.

    Args:
        n_arms: Number of bandit arms.
        mu0: Prior mean (default 0.0).
        lam: Prior pseudo-count (default 1.0).
        alpha: Prior shape (default 1.0).
        beta: Prior rate (default 1.0).
        device: Torch device.
    """

    def __init__(
        self,
        n_arms: int,
        mu0: float = 0.0,
        lam: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.n_arms = n_arms
        self.device = device

        # Prior hyper-parameters (stored per arm)
        self._mu0 = torch.full((n_arms,), mu0, device=device, dtype=torch.float64)
        self._lam = torch.full((n_arms,), lam, device=device, dtype=torch.float64)
        self._alpha = torch.full((n_arms,), alpha, device=device, dtype=torch.float64)
        self._beta = torch.full((n_arms,), beta, device=device, dtype=torch.float64)

        # Remember initial values for reset
        self._init_mu0 = mu0
        self._init_lam = lam
        self._init_alpha = alpha
        self._init_beta = beta

    # ------------------------------------------------------------------
    # Posterior update
    # ------------------------------------------------------------------

    def update(self, arm: int, observation: float) -> None:
        """Bayesian update of a single arm's NIG posterior.

        Args:
            arm: Arm index to update.
            observation: Observed reward value.
        """
        mu0 = self._mu0[arm]
        lam = self._lam[arm]
        alpha = self._alpha[arm]
        beta = self._beta[arm]

        lam_new = lam + 1.0
        mu0_new = (lam * mu0 + observation) / lam_new
        alpha_new = alpha + 0.5
        beta_new = beta + 0.5 * lam * (observation - mu0) ** 2 / lam_new

        self._mu0[arm] = mu0_new
        self._lam[arm] = lam_new
        self._alpha[arm] = alpha_new
        self._beta[arm] = beta_new

    def update_batch(self, arms: Tensor, observations: Tensor) -> None:
        """Bayesian update for a batch of (arm, observation) pairs.

        Args:
            arms: Int tensor of arm indices, shape (B,).
            observations: Float tensor of rewards, shape (B,).
        """
        for a, x in zip(arms.tolist(), observations.tolist()):
            self.update(int(a), float(x))

    # ------------------------------------------------------------------
    # Thompson Sampling
    # ------------------------------------------------------------------

    def sample(self) -> tuple[Tensor, Tensor]:
        """Draw (mean, variance) from the NIG posterior for every arm.

        Returns:
            mean: Sampled means, shape (n_arms,).
            variance: Sampled variances, shape (n_arms,).
        """
        # Sample variance ~ InverseGamma(α, β)
        # InverseGamma(α, β) = 1 / Gamma(α, 1/β)
        # PyTorch Gamma uses concentration=α, rate=1/β → shape/rate param
        gamma_samples = torch.distributions.Gamma(
            self._alpha, 1.0 / self._beta
        ).sample()
        variance = 1.0 / gamma_samples.clamp(min=1e-10)

        # Sample mean ~ Normal(μ₀, variance / λ)
        std = (variance / self._lam).sqrt()
        mean = torch.distributions.Normal(self._mu0, std).sample()

        return mean.float(), variance.float()

    def sample_means(self) -> Tensor:
        """Draw posterior mean samples for every arm (convenience wrapper).

        Returns:
            Sampled means, shape (n_arms,).
        """
        mean, _ = self.sample()
        return mean

    def select_arm(self) -> int:
        """Select the arm with the highest sampled mean."""
        return int(self.sample_means().argmax().item())

    def sample_batch_means(self, n: int) -> Tensor:
        """Draw n independent posterior mean samples for every arm.

        Vectorizes the sampling so that n draws share a single
        distribution construction rather than n separate calls.

        Args:
            n: Number of independent samples to draw.

        Returns:
            Sampled means, shape (n, n_arms).
        """
        # Expand parameters: (n_arms,) -> (n, n_arms)
        alpha = self._alpha.unsqueeze(0).expand(n, -1)
        beta = self._beta.unsqueeze(0).expand(n, -1)
        lam = self._lam.unsqueeze(0).expand(n, -1)
        mu0 = self._mu0.unsqueeze(0).expand(n, -1)

        # Sample variance ~ InverseGamma(α, β) for each of n draws
        gamma_samples = torch.distributions.Gamma(alpha, 1.0 / beta).sample()
        variance = 1.0 / gamma_samples.clamp(min=1e-10)

        # Sample mean ~ Normal(μ₀, variance / λ)
        std = (variance / lam).sqrt()
        mean = torch.distributions.Normal(mu0, std).sample()

        return mean.float()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def posterior_mean(self) -> Tensor:
        """Point-estimate posterior means for all arms."""
        return self._mu0.float()

    @property
    def posterior_count(self) -> Tensor:
        """Effective observation count per arm (λ − init_λ)."""
        return (self._lam - self._init_lam).float()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all arms to the initial prior."""
        self._mu0.fill_(self._init_mu0)
        self._lam.fill_(self._init_lam)
        self._alpha.fill_(self._init_alpha)
        self._beta.fill_(self._init_beta)

    def clone(self) -> NIGStats:
        """Return an independent copy of this stats object."""
        new = NIGStats(
            n_arms=self.n_arms,
            mu0=self._init_mu0,
            lam=self._init_lam,
            alpha=self._init_alpha,
            beta=self._init_beta,
            device=self.device,
        )
        new._mu0 = self._mu0.clone()
        new._lam = self._lam.clone()
        new._alpha = self._alpha.clone()
        new._beta = self._beta.clone()
        return new
