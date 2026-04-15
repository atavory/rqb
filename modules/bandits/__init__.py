#!/usr/bin/env python3


"""Bandit modules for hierarchical Thompson Sampling with RQ trunk addresses.

This package provides infrastructure for contextual bandits that leverage
trunk addresses from RQ as discrete context representations, enabling:
    - O(1) decision time via lookup instead of neural network inference
    - Bayesian uncertainty via Normal-Inverse-Gamma conjugate priors
    - Cold-start transfer from pre-trained classifiers

Key components:
    - NIGStats: Normal-Inverse-Gamma sufficient statistics for each trunk address
    - HierarchicalThompsonSampling: Bandit policy using trunk-indexed priors
    - ColdStartTransfer: Transfer learning from classifier to bandit
"""

from modules.bandits.cold_start import ColdStartTransfer
from modules.bandits.hierarchical_ts import (
    BanditConfig,
    ExplorationStrategy,
    HierarchicalThompsonSampling,
    NeuralTSBaseline,
)
from modules.bandits.nig_stats import NIGStats

__all__ = [
    "NIGStats",
    "HierarchicalThompsonSampling",
    "NeuralTSBaseline",
    "BanditConfig",
    "ExplorationStrategy",
    "ColdStartTransfer",
]
