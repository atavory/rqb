

"""Encoder architectures for contrastive learning experiments.

This module provides encoder implementations for different modalities:
    - TabTransformer: Transformer-based encoder for tabular data
    - ResNetEncoder: ResNet-18 for image data (CIFAR experiments)
    - SCARFEncoder: Self-supervised MLP encoder via random feature corruption
"""

from modules.encoders.resnet import ResNetEncoder
from modules.encoders.scarf import SCARFEncoder
from modules.encoders.tab_transformer import TabTransformer

__all__ = [
    "TabTransformer",
    "ResNetEncoder",
    "SCARFEncoder",
]
