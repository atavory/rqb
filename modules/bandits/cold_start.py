

"""Cold-start transfer for hierarchical bandits.

Transfers NIG priors from one context to another so that a bandit policy
can bootstrap new trunk addresses using statistics from previously-seen
(similar) addresses. This is primarily used in P3 experiments but is
exported from the bandits package for completeness.
"""

from __future__ import annotations

from modules.bandits.nig_stats import NIGStats


class ColdStartTransfer:
    """Transfer NIG priors between trunk-address contexts.

    Given a source context's NIGStats, produce an initialised NIGStats for
    a target context that has never been seen (cold start).

    The simplest strategy is to copy the source prior directly; more
    sophisticated approaches (shrinkage, hierarchical pooling) can be
    added later.

    Args:
        n_arms: Number of bandit arms.
        shrinkage: Blend factor toward the global prior (0 = full copy,
            1 = ignore source and use uninformative prior).
        device: Torch device.
    """

    def __init__(
        self,
        n_arms: int,
        shrinkage: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.n_arms = n_arms
        self.shrinkage = shrinkage
        self.device = device

    def transfer(
        self,
        source: NIGStats,
        mu0_prior: float = 0.0,
        lam_prior: float = 1.0,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> NIGStats:
        """Create a new NIGStats for a cold-start context.

        When shrinkage == 0 the returned stats are an exact copy of
        *source*.  When shrinkage == 1 the returned stats equal the
        uninformative prior specified by the ``*_prior`` arguments.
        Intermediate values linearly interpolate the NIG parameters.

        Args:
            source: NIGStats from a previously-seen context.
            mu0_prior: Global prior mean.
            lam_prior: Global prior pseudo-count.
            alpha_prior: Global prior shape.
            beta_prior: Global prior rate.

        Returns:
            A new NIGStats blending *source* and the global prior.
        """
        s = self.shrinkage
        target = NIGStats(
            n_arms=self.n_arms,
            mu0=mu0_prior,
            lam=lam_prior,
            alpha=alpha_prior,
            beta=beta_prior,
            device=self.device,
        )

        if s < 1.0:
            # Blend source posterior into the fresh prior
            target._mu0 = (1.0 - s) * source._mu0 + s * target._mu0
            target._lam = (1.0 - s) * source._lam + s * target._lam
            target._alpha = (1.0 - s) * source._alpha + s * target._alpha
            target._beta = (1.0 - s) * source._beta + s * target._beta

        return target
