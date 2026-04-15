

"""Hierarchical Thompson Sampling using RQ trunk addresses.

Provides:
    - ExplorationStrategy enum (TS / UCB / ε-Greedy)
    - BanditConfig dataclass
    - HierarchicalThompsonSampling — O(1) bandit using trunk codes as context
    - NeuralTSBaseline — MC-Dropout neural network baseline (used for latency comparison)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch
from modules.bandits.nig_stats import NIGStats
from torch import nn, Tensor


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------


class ExplorationStrategy(enum.Enum):
    """Exploration strategy for the bandit."""

    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"

    def __str__(self) -> str:
        return self.value


@dataclass
class BanditConfig:
    """Configuration for the bandit policy.

    Args:
        exploration: Which exploration strategy to use.
        epsilon: ε for ε-Greedy (ignored otherwise).
        ucb_confidence: Confidence multiplier for UCB (ignored otherwise).
        mu0: NIG prior mean.
        lam: NIG prior pseudo-count.
        alpha: NIG prior shape.
        beta: NIG prior rate.
    """

    exploration: ExplorationStrategy = ExplorationStrategy.THOMPSON_SAMPLING
    epsilon: float = 0.1
    ucb_confidence: float = 2.0
    mu0: float = 0.0
    lam: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0


# -----------------------------------------------------------------------
# Hierarchical Thompson Sampling
# -----------------------------------------------------------------------


class HierarchicalThompsonSampling:
    """Contextual bandit using RQ trunk codes as discrete context keys.

    Each unique trunk-code tuple maps to an independent set of per-arm NIG
    priors. Decision time is O(1) dictionary lookup + NIG posterior sample.

    Args:
        n_arms: Number of bandit arms.
        n_codes: Codebook size per RQ level (used for documentation only).
        d_cut: Number of trunk levels whose codes form the context key.
        config: Optional exploration / prior configuration.
        device: Torch device.
    """

    def __init__(
        self,
        n_arms: int,
        n_codes: int,
        d_cut: int,
        config: BanditConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.n_arms = n_arms
        self.n_codes = n_codes
        self.d_cut = d_cut
        self.config = config or BanditConfig()
        self.device = device

        # trunk_code_tuple → NIGStats
        self._contexts: dict[tuple[int, ...], NIGStats] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _codes_to_key(self, codes: Tensor) -> tuple[int, ...]:
        """Convert a single row of trunk codes to a hashable tuple."""
        return tuple(codes.tolist())

    def _get_or_create_stats(self, key: tuple[int, ...]) -> NIGStats:
        """Retrieve the NIGStats for a context key, creating if needed."""
        if key not in self._contexts:
            self._contexts[key] = NIGStats(
                n_arms=self.n_arms,
                mu0=self.config.mu0,
                lam=self.config.lam,
                alpha=self.config.alpha,
                beta=self.config.beta,
                device=self.device,
            )
        return self._contexts[key]

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def select_arm(self, codes: Tensor) -> Tensor:
        """Select an arm for each sample in the batch.

        Groups batch items by context key and vectorizes arm selection
        within each group to reduce Python loop overhead.

        Args:
            codes: Trunk codes, shape (B, d_cut).

        Returns:
            Selected arm indices, shape (B,).
        """
        batch_size = codes.shape[0]
        arms = torch.empty(batch_size, dtype=torch.long, device=self.device)

        # Group batch indices by context key
        key_to_indices: dict[tuple[int, ...], list[int]] = {}
        for i in range(batch_size):
            key = self._codes_to_key(codes[i])
            if key not in key_to_indices:
                key_to_indices[key] = []
            key_to_indices[key].append(i)

        # Process each unique context once
        for key, indices in key_to_indices.items():
            stats = self._get_or_create_stats(key)
            n = len(indices)

            if self.config.exploration == ExplorationStrategy.THOMPSON_SAMPLING:
                # Sample n sets of arm means in one batch
                sampled = stats.sample_batch_means(n)  # (n, n_arms)
                selected = sampled.argmax(dim=-1)  # (n,)
                for j, idx in enumerate(indices):
                    arms[idx] = selected[j]

            elif self.config.exploration == ExplorationStrategy.UCB:
                mean = stats.posterior_mean
                count = stats.posterior_count
                total = count.sum().clamp(min=1.0)
                bonus = (
                    self.config.ucb_confidence
                    * (torch.log(total) / (count + 1.0)).sqrt()
                )
                best = (mean + bonus).argmax()
                for idx in indices:
                    arms[idx] = best

            elif self.config.exploration == ExplorationStrategy.EPSILON_GREEDY:
                best = stats.posterior_mean.argmax()
                for idx in indices:
                    if torch.rand(1).item() < self.config.epsilon:
                        arms[idx] = torch.randint(self.n_arms, (1,)).item()
                    else:
                        arms[idx] = best

        return arms

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        codes: Tensor,
        arms: Tensor,
        rewards: Tensor,
    ) -> None:
        """Bayesian posterior update for a batch of observations.

        Args:
            codes: Trunk codes, shape (B, d_cut).
            arms: Chosen arm indices, shape (B,).
            rewards: Observed rewards, shape (B,).
        """
        for i in range(codes.shape[0]):
            key = self._codes_to_key(codes[i])
            stats = self._get_or_create_stats(key)
            stats.update(int(arms[i].item()), float(rewards[i].item()))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_contexts_seen(self) -> int:
        """Number of unique trunk-code contexts encountered so far."""
        return len(self._contexts)


# -----------------------------------------------------------------------
# Neural Thompson Sampling Baseline (latency comparison only)
# -----------------------------------------------------------------------


class NeuralTSBaseline(nn.Module):
    """MC-Dropout neural network baseline for Thompson Sampling.

    Used for latency benchmarks (run_nts_experiment.py) to demonstrate
    the O(1) advantage of HierarchicalThompsonSampling. Not used in the
    main real-data bandit experiments (run_real_nts_experiment.py) because
    it consistently underperforms even Random.

    Args:
        input_dim: Embedding dimension.
        hidden_dim: Hidden layer width.
        n_arms: Number of bandit arms.
        n_mc_samples: Number of MC forward passes for uncertainty.
        dropout: Dropout probability.
        lr: Learning rate for online updates.
        buffer_size: Maximum replay buffer size.
        train_every: Train on the buffer every N observations.
        train_steps: Number of SGD steps per training round.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_arms: int,
        n_mc_samples: int = 20,
        dropout: float = 0.1,
        lr: float = 1e-3,
        buffer_size: int = 10000,
        train_every: int = 50,
        train_steps: int = 10,
    ) -> None:
        super().__init__()
        self.n_arms = n_arms
        self.n_mc_samples = n_mc_samples
        self.train_every = train_every
        self.train_steps = train_steps
        self.buffer_size = buffer_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_arms),
        )

        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._buffer_x: list[Tensor] = []
        self._buffer_arm: list[int] = []
        self._buffer_reward: list[float] = []
        self._n_updates: int = 0

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning arm values, shape (B, n_arms)."""
        return self.net(x)

    @torch.no_grad()
    def select_arm(self, embeddings: Tensor) -> Tensor:
        """Select arms via MC-Dropout Thompson Sampling.

        Args:
            embeddings: Input embeddings, shape (B, input_dim).

        Returns:
            Selected arm indices, shape (B,).
        """
        self.train()  # Enable dropout for MC sampling
        accum = torch.zeros(embeddings.shape[0], self.n_arms, device=embeddings.device)
        for _ in range(self.n_mc_samples):
            accum += self.forward(embeddings)
        mean_values = accum / self.n_mc_samples
        return mean_values.argmax(dim=-1)

    def update(self, embedding: Tensor, arm: int, reward: float) -> None:
        """Store observation and periodically retrain on the replay buffer."""
        self._buffer_x.append(embedding.detach().reshape(1, -1))
        self._buffer_arm.append(arm)
        self._buffer_reward.append(reward)

        if len(self._buffer_x) > self.buffer_size:
            self._buffer_x.pop(0)
            self._buffer_arm.pop(0)
            self._buffer_reward.pop(0)

        self._n_updates += 1
        if self._n_updates % self.train_every == 0 and len(self._buffer_x) > 0:
            self._train_on_buffer()

    def _train_on_buffer(self) -> None:
        """Run a few SGD steps on the replay buffer (MSE on chosen arm)."""
        x = torch.cat(self._buffer_x, dim=0)
        arms = torch.tensor(self._buffer_arm, dtype=torch.long, device=x.device)
        rewards = torch.tensor(
            self._buffer_reward, dtype=torch.float32, device=x.device
        )

        self.train()
        for _ in range(self.train_steps):
            preds = self.forward(x)  # (N, n_arms)
            pred_for_arm = preds.gather(1, arms.unsqueeze(1)).squeeze(1)
            loss = torch.nn.functional.mse_loss(pred_for_arm, rewards)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
