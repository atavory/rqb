

"""TabTransformer encoder for tabular data.

Implementation based on "TabTransformer: Tabular Data Modeling Using Contextual
Embeddings" (Huang et al., 2020). Designed for SSL experiments on OpenML datasets.

The encoder:
    1. Embeds categorical features via learned embeddings
    2. Projects continuous features to embedding dimension
    3. Applies transformer layers for contextual mixing
    4. Outputs a single embedding vector via CLS token or mean pooling

Example:
    >>> encoder = TabTransformer(
    ...     n_categories=[10, 5, 20],  # Cardinalities for 3 categorical features
    ...     n_continuous=4,
    ...     dim=64,
    ...     n_heads=4,
    ...     n_layers=3,
    ... )
    >>> cat_features = torch.randint(0, 10, (32, 3))  # 32 samples, 3 categorical
    >>> cont_features = torch.randn(32, 4)  # 32 samples, 4 continuous
    >>> embeddings = encoder(cat_features, cont_features)  # (32, 64)
"""

from __future__ import annotations

import torch
from torch import nn, Tensor


class TabTransformer(nn.Module):
    """TabTransformer encoder for tabular data.

    Attributes:
        n_categories: List of category counts for each categorical feature.
        n_continuous: Number of continuous features.
        dim: Embedding/hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
    """

    def __init__(
        self,
        n_categories: list[int],
        n_continuous: int,
        dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        """Initialize TabTransformer.

        Args:
            n_categories: List of cardinalities for each categorical feature.
            n_continuous: Number of continuous features.
            dim: Hidden dimension for embeddings and transformer.
            n_heads: Number of attention heads.
            n_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
            use_cls_token: If True, use CLS token for output; else mean pool.
        """
        super().__init__()

        self.n_categories = n_categories
        self.n_continuous = n_continuous
        self.dim = dim
        self.use_cls_token = use_cls_token

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, dim) for n_cat in n_categories]
        )

        if n_continuous > 0:
            self.cont_projection = nn.Linear(1, dim)
            self.cont_bn = nn.BatchNorm1d(n_continuous)
        else:
            self.cont_projection = None
            self.cont_bn = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        n_tokens = len(n_categories) + n_continuous + (1 if use_cls_token else 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings and projections."""
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

        if self.cont_projection is not None:
            nn.init.xavier_uniform_(self.cont_projection.weight)
            nn.init.zeros_(self.cont_projection.bias)

        nn.init.normal_(self.pos_embedding, std=0.02)

        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        cat_features: Tensor | None = None,
        cont_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            cat_features: Categorical features, shape (batch_size, n_cat_features).
                Values should be integer indices in [0, n_categories[i]).
            cont_features: Continuous features, shape (batch_size, n_continuous).

        Returns:
            Embeddings, shape (batch_size, dim).
        """
        batch_size = (
            cat_features.shape[0]
            if cat_features is not None
            else cont_features.shape[0]
        )

        tokens: list[Tensor] = []

        if cat_features is not None:
            for i, emb in enumerate(self.cat_embeddings):
                tokens.append(emb(cat_features[:, i]))

        if cont_features is not None and self.cont_projection is not None:
            cont_normalized = self.cont_bn(cont_features)
            for i in range(self.n_continuous):
                proj = self.cont_projection(cont_normalized[:, i : i + 1])
                tokens.append(proj)

        x = torch.stack(tokens, dim=1)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embedding[:, : x.shape[1], :]

        x = self.transformer(x)

        if self.use_cls_token:
            return x[:, 0, :]
        else:
            return x.mean(dim=1)


class MLPEncoder(nn.Module):
    """Simple MLP encoder for tabular data (baseline).

    A straightforward feedforward network for comparison with TabTransformer.
    """

    def __init__(
        self,
        n_categories: list[int],
        n_continuous: int,
        dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize MLP encoder.

        Args:
            n_categories: List of cardinalities for each categorical feature.
            n_continuous: Number of continuous features.
            dim: Output embedding dimension.
            hidden_dims: Hidden layer dimensions. Defaults to [128, 64].
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_categories = n_categories
        self.n_continuous = n_continuous

        cat_embed_dim = 8
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, cat_embed_dim) for n_cat in n_categories]
        )

        input_dim = len(n_categories) * cat_embed_dim + n_continuous

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, dim))

        self.mlp = nn.Sequential(*layers)

        if n_continuous > 0:
            self.cont_bn = nn.BatchNorm1d(n_continuous)
        else:
            self.cont_bn = None

    def forward(
        self,
        cat_features: Tensor | None = None,
        cont_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            cat_features: Categorical features, shape (batch_size, n_cat_features).
            cont_features: Continuous features, shape (batch_size, n_continuous).

        Returns:
            Embeddings, shape (batch_size, dim).
        """
        parts = []

        if cat_features is not None:
            cat_embeds = [
                emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)
            ]
            parts.append(torch.cat(cat_embeds, dim=-1))

        if cont_features is not None and self.cont_bn is not None:
            parts.append(self.cont_bn(cont_features))

        x = torch.cat(parts, dim=-1)
        return self.mlp(x)
