# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""XGBoost Epsilon-Greedy contextual bandit.

A single global XGBoost model predicts rewards for (context, arm) pairs.
Exploration is handled via epsilon-greedy with a log-decaying schedule:
    epsilon_t = epsilon_0 / log(t + 2)

The model is retrained periodically on the full replay buffer (batch-fit).
"""

from __future__ import annotations

import numpy as np


class XGBGreedyBaseline:
    def __init__(
        self,
        input_dim: int,
        n_arms: int,
        epsilon: float = 0.1,
        retrain_every: int = 100,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.n_arms = n_arms
        self.d = input_dim
        self.epsilon_init = epsilon
        self.retrain_every = retrain_every
        self.rng = rng or np.random.RandomState()
        self.step_count = 0

        self.model: object | None = None
        self.replay_x: list[np.ndarray] = []
        self.replay_y: list[float] = []

    def _make_feature(self, context: np.ndarray, arm: int) -> np.ndarray:
        """Concatenate context with one-hot arm indicator."""
        x = context.flatten()
        arm_oh = np.zeros(self.n_arms)
        arm_oh[arm] = 1.0
        return np.concatenate([x, arm_oh])

    def select_arm(self, context: np.ndarray) -> int:
        self.step_count += 1
        eps_t = self.epsilon_init / np.log(self.step_count + 1)

        if self.rng.rand() < eps_t:
            return self.rng.randint(self.n_arms)

        if self.model is None:
            return self.rng.randint(self.n_arms)

        all_arm_feats = np.array(
            [self._make_feature(context, a) for a in range(self.n_arms)]
        )
        preds = self.model.predict(all_arm_feats)  # pyre-ignore[16]
        return int(np.argmax(preds))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        self.replay_x.append(self._make_feature(context, arm))
        self.replay_y.append(reward)

        if self.step_count > 0 and self.step_count % self.retrain_every == 0:
            self._retrain()

    def _retrain(self) -> None:
        """Fit XGBoost on the entire replay buffer."""
        import xgboost as xgb  # lazy import — only needed if method is active

        X = np.array(self.replay_x)
        y = np.array(self.replay_y)
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            verbosity=0,
            n_jobs=1,
            random_state=self.rng.randint(2**31),
        )
        self.model.fit(X, y)
