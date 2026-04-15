#!/usr/bin/env python3


"""Projection head for contrastive learning."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for contrastive learning.

    Maps encoder embeddings to a lower-dimensional space where the
    contrastive loss is computed. This follows the SimCLR/MoCo convention
    of using a non-linear projection head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(self.net(x), dim=1)
