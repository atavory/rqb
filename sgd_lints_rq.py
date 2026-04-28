# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""SGD-LinTS-RQ bandit with adaptive depth selection.

Diagonal-precision Linear Thompson Sampling over a residual quantization
(RQ) hierarchy.  Each level of the codebook gets one diagonal-precision
LinTS model per centroid per arm.  Scores are summed across active levels
with η-damping.  A shadow level one step deeper is trained passively; a
paired t-test on windowed prediction error decides whether to promote.
O(ld) per step where l is the active depth and d is the feature dimension.
"""

from __future__ import annotations

import numpy as np

from shadow_test import freedman_promotion, ttest_promotion


class SGDLinDRQm:
    def __init__(
        self,
        feat_dim: int,
        n_arms: int,
        b_per_level: int,
        max_depth: int = 8,
        min_level: int = 1,
        max_level: int = 8,
        nu: float = 0.1,
        lam: float = 1.0,
        eta: float = 0.5,
        c: float = 50.0,
        W: int = 1000,
        alpha: float = 0.05,
        seed: int = 0,
        promotion_test: str = "ttest",
    ) -> None:
        self.n_arms = n_arms
        self.b_per_level = b_per_level
        self.max_depth = max_depth
        self.min_level = min_level
        self.max_level = min(max_level, max_depth)
        self.nu = nu
        self.eta = eta
        self.W = W
        self.promotion_test = promotion_test
        self.feat_dim = feat_dim + 1  # +1 for intercept
        self.rng = np.random.RandomState(seed)

        T_0 = int(c * b_per_level * n_arms)
        self.max_checks = max_depth * 3
        self.alpha_corrected = alpha / self.max_checks
        self.check_times: set[int] = set()
        t_check = T_0
        for _ in range(self.max_checks):
            self.check_times.add(t_check)
            t_check *= 2

        self.active_depth = min_level
        self.shadow_depth = min_level + 1

        self.prec = [
            [
                [np.full(self.feat_dim, lam, dtype=np.float64) for _ in range(n_arms)]
                for _ in range(b_per_level)
            ]
            for _ in range(max_depth)
        ]
        self.f = [
            [
                [np.zeros(self.feat_dim, dtype=np.float64) for _ in range(n_arms)]
                for _ in range(b_per_level)
            ]
            for _ in range(max_depth)
        ]

        self.error_buffer_active = np.zeros(W, dtype=np.float64)
        self.error_buffer_shadow = np.zeros(W, dtype=np.float64)
        self.error_buf_idx = 0
        self.error_buf_count = 0

        self.t = 0
        self.promotion_log: list[tuple[int, int, int]] = []

    def _add_intercept(self, x: np.ndarray) -> np.ndarray:
        return np.append(x, 1.0)

    def select_arm(self, trunk_codes: np.ndarray, residual_feats: np.ndarray) -> int:
        best_arm = 0
        best_val = -float("inf")
        for a in range(self.n_arms):
            z = 0.0
            for lvl in range(self.active_depth):
                c = int(trunk_codes[lvl])
                r = self._add_intercept(residual_feats[lvl])
                pr = self.prec[lvl][c][a]
                mu = self.f[lvl][c][a] / pr
                sigma = self.nu / np.sqrt(pr)
                theta_sample = mu + sigma * self.rng.randn(self.feat_dim)
                z += (self.eta**lvl) * float(theta_sample @ r)
            if z > best_val:
                best_val = z
                best_arm = a
        return best_arm

    def _mean_for_arm(
        self,
        trunk_codes: np.ndarray,
        residual_feats: np.ndarray,
        arm: int,
        depth: int,
    ) -> float:
        z = 0.0
        for lvl in range(depth):
            c = int(trunk_codes[lvl])
            r = self._add_intercept(residual_feats[lvl])
            mu = self.f[lvl][c][arm] / self.prec[lvl][c][arm]
            z += (self.eta**lvl) * float(mu @ r)
        return z

    def update(
        self,
        trunk_codes: np.ndarray,
        residual_feats: np.ndarray,
        arm: int,
        reward: float,
    ) -> None:
        pred_active = np.clip(
            self._mean_for_arm(trunk_codes, residual_feats, arm, self.active_depth)
            + 0.5,
            0.0,
            1.0,
        )
        shadow_up_to = min(self.shadow_depth, self.max_level)
        pred_shadow = np.clip(
            self._mean_for_arm(trunk_codes, residual_feats, arm, shadow_up_to) + 0.5,
            0.0,
            1.0,
        )

        sq_err_active = (reward - pred_active) ** 2
        sq_err_shadow = (reward - pred_shadow) ** 2
        buf_pos = self.error_buf_idx % self.W
        self.error_buffer_active[buf_pos] = sq_err_active
        self.error_buffer_shadow[buf_pos] = sq_err_shadow
        self.error_buf_idx += 1
        self.error_buf_count = min(self.error_buf_count + 1, self.W)

        cumulative_logit_active = 0.0
        for lvl in range(self.active_depth):
            c = int(trunk_codes[lvl])
            r = self._add_intercept(residual_feats[lvl])
            target = reward - np.clip(cumulative_logit_active + 0.5, 0.0, 1.0)
            mu_pre = self.f[lvl][c][arm] / self.prec[lvl][c][arm]
            pre_pred = float(mu_pre @ r)
            self.prec[lvl][c][arm] += r * r
            self.f[lvl][c][arm] += target * r
            cumulative_logit_active += (self.eta**lvl) * pre_pred

        if shadow_up_to <= self.max_level:
            shadow_lvl = self.shadow_depth - 1
            if shadow_lvl < len(trunk_codes):
                c = int(trunk_codes[shadow_lvl])
                r = self._add_intercept(residual_feats[shadow_lvl])
                target = reward - np.clip(cumulative_logit_active + 0.5, 0.0, 1.0)
                self.prec[shadow_lvl][c][arm] += r * r
                self.f[shadow_lvl][c][arm] += target * r

        self.t += 1

        if self.t in self.check_times and self.shadow_depth <= self.max_level:
            self._maybe_promote()

    def _maybe_promote(self) -> None:
        if self.error_buf_count < self.W:
            return

        n = self.error_buf_count
        if self.promotion_test == "freedman":
            promote = freedman_promotion(
                self.error_buffer_active, self.error_buffer_shadow,
                n, self.alpha_corrected,
            )
        else:
            promote = ttest_promotion(
                self.error_buffer_active, self.error_buffer_shadow,
                n, self.alpha_corrected,
            )

        if promote:
            old_depth = self.active_depth
            self.active_depth = self.shadow_depth
            self.shadow_depth = self.active_depth + 1
            self.promotion_log.append((self.t, old_depth, self.active_depth))
            self.error_buffer_active[:] = 0.0
            self.error_buffer_shadow[:] = 0.0
            self.error_buf_idx = 0
            self.error_buf_count = 0
