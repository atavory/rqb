

"""ResNet encoder for image data (CIFAR experiments).

Standard ResNet-18 modified for smaller images (32x32 CIFAR) and SSL.
Removes the classification head, outputs embedding vector.

Example:
    >>> encoder = ResNetEncoder(dim=128)
    >>> images = torch.randn(32, 3, 32, 32)  # CIFAR-sized
    >>> embeddings = encoder(images)  # (32, 128)
"""

from __future__ import annotations

import torch
from torch import nn, Tensor


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder for image data.

    Modified for CIFAR-sized images (32x32):
    - Uses 3x3 initial conv instead of 7x7
    - Removes initial max pooling

    Attributes:
        dim: Output embedding dimension.
    """

    def __init__(
        self,
        dim: int = 128,
        in_channels: int = 3,
        use_projection_head: bool = True,
        projection_hidden_dim: int = 256,
    ) -> None:
        """Initialize ResNet encoder.

        Args:
            dim: Output embedding dimension.
            in_channels: Number of input channels (3 for RGB).
            use_projection_head: If True, add MLP projection head (for SSL).
            projection_hidden_dim: Hidden dim of projection head.
        """
        super().__init__()

        self.dim = dim
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if use_projection_head:
            self.projection = nn.Sequential(
                nn.Linear(512, projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_dim, dim),
            )
        else:
            self.projection = nn.Linear(512, dim)

        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input images, shape (batch_size, channels, height, width).

        Returns:
            Embeddings, shape (batch_size, dim).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.projection(x)

        return x

    def get_features(self, x: Tensor) -> Tensor:
        """Get features before projection head.

        Args:
            x: Input images, shape (batch_size, channels, height, width).

        Returns:
            Features, shape (batch_size, 512).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
