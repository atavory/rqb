# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Shadow promotion tests: t-test and Freedman's inequality."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ttest_promotion(
    errors_active: np.ndarray,
    errors_shadow: np.ndarray,
    n: int,
    alpha_corrected: float,
) -> bool:
    """Paired t-test: returns True if shadow is significantly better."""
    diffs = errors_active[:n] - errors_shadow[:n]
    if float(np.mean(diffs)) <= 0:
        return False
    _t_stat, p_two = stats.ttest_rel(errors_active[:n], errors_shadow[:n])
    return (p_two / 2.0) < alpha_corrected


def freedman_promotion(
    errors_active: np.ndarray,
    errors_shadow: np.ndarray,
    n: int,
    alpha_corrected: float,
) -> bool:
    """Freedman's inequality test: returns True if shadow is significantly better.

    Tests whether the sum of paired differences D_i = err_active_i - err_shadow_i
    is significantly positive, using Freedman's martingale inequality
    (valid for sequential, non-i.i.d. data).

    Freedman's inequality: P(S_n >= eps AND V_n <= v) <= exp(-eps^2 / (2v + 2b*eps/3))
    where S_n = sum(D_i - E[D_i|F_{i-1}]), V_n = sum(Var[D_i|F_{i-1}]),
    b = max|D_i|.

    For our application with bounded rewards in [0,1], |D_i| <= 1, so b = 1.
    We use the empirical sum and variance as proxies.
    """
    diffs = errors_active[:n] - errors_shadow[:n]
    S_n = float(np.sum(diffs))
    if S_n <= 0:
        return False

    # Empirical variance proxy for the conditional variance sum
    V_n = float(np.sum((diffs - np.mean(diffs)) ** 2))
    b = float(np.max(np.abs(diffs)))
    if b == 0:
        return False

    # Freedman bound: P(S_n >= eps) <= exp(-eps^2 / (2*V_n + 2*b*eps/3))
    # We invert: reject if exp(-S_n^2 / (2*V_n + 2*b*S_n/3)) < alpha
    denom = 2.0 * V_n + 2.0 * b * S_n / 3.0
    if denom <= 0:
        return S_n > 0

    log_p = -(S_n ** 2) / denom
    return log_p < np.log(alpha_corrected)
