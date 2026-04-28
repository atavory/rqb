# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Doubling XGBoost Epsilon-Greedy contextual bandit.

Epoch-doubling schedule: starts with ``init_phase`` rounds of random
exploration, then fits an XGBoost regressor on (context||arm_onehot) → reward.
Each subsequent phase doubles in length (500 → 1000 → 2000 → …).  At each
phase boundary the model is retrained on the full replay buffer.

Within a phase the agent plays epsilon-greedy (fixed epsilon, default 0.05)
using the current model.
"""

from __future__ import annotations

import logging

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class DoublingXGBGreedy:
    """XGBoost epsilon-greedy bandit with doubling-epoch retraining."""

    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        epsilon: float = 0.05,
        init_phase: int = 500,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.epsilon = epsilon
        self.init_phase = init_phase
        self.rng = rng or np.random.RandomState()

        self.step_count = 0
        self.phase_len = init_phase
        self.next_train_at = init_phase

        self.model: object | None = None
        self.replay_x: list[np.ndarray] = []
        self.replay_y: list[float] = []

    def _make_feature(self, context: np.ndarray, arm: int) -> np.ndarray:
        x = context.flatten()
        arm_oh = np.zeros(self.n_arms)
        arm_oh[arm] = 1.0
        return np.concatenate([x, arm_oh])

    def select_arm(self, context: np.ndarray) -> int:
        if self.model is None or self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.n_arms))

        all_arm_feats = np.array(
            [self._make_feature(context, a) for a in range(self.n_arms)]
        )
        preds = self.model.predict(all_arm_feats)  # pyre-ignore[16]
        return int(np.argmax(preds))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        self.replay_x.append(self._make_feature(context, arm))
        self.replay_y.append(reward)
        self.step_count += 1

        if self.step_count >= self.next_train_at:
            self._retrain()
            self.phase_len *= 2
            self.next_train_at = self.step_count + self.phase_len

    def _retrain(self) -> None:
        import xgboost as xgb  # pyre-ignore[21]

        X = np.array(self.replay_x)
        y = np.array(self.replay_y)
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            verbosity=0,
            n_jobs=-1,
            random_state=self.rng.randint(2**31),
        )
        self.model.fit(X, y)
