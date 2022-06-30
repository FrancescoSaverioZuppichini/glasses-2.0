from typing import List

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from ..base import Backbone


class ViTTokens(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def __len__(self):
        return len(list(self.parameters()))

    def forward(self, x: Tensor) -> List[Tensor]:
        b = x.shape[0]
        tokens = []
        for token in self.parameters():
            tokens.append(repeat(token, "() n e -> b n e", b=b))
        return tokens


class ViTPatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        img_size: int = 224,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.tokens = ViTTokens(embed_dim)
        self.positions = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + len(self.tokens), embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        tokens = self.tokens(x)
        x = torch.cat([*tokens, x], dim=1)
        x = x + self.positions
        return x


class ViTAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop_p: float = 0.0,
        projection_drop_p: float = 0.2,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.att_drop = nn.Dropout(attn_drop_p)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(projection_drop_p)
        )

        self.scaling = (self.embed_dim // num_heads) ** -0.5

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(
            self.qkv(x), "b n (qkv h d) -> (qkv) b h n d", h=self.num_heads, qkv=3
        )

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # dot product, Q V^T, here we don't transpose before, so this is why
        # the sum is made on the last index of  K
        energy = torch.einsum("bhij, bhkj -> bhik", queries, keys) * self.scaling
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        # dot product
        out = torch.einsum("bhij, bhjk -> bhik ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ViTMLPBlock(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        expansion: int = 4,
        drop_p: float = 0.0,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__(
            nn.Linear(embed_dim, expansion * embed_dim),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embed_dim, embed_dim),
            nn.Dropout(drop_p),
        )


class ResidualAddition(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop_p: float = 0.0,
        projection_drop_p: float = 0.2,
        qkv_bias: bool = False,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.2,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.transformer = ResidualAddition(
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                ViTAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop_p=attn_drop_p,
                    projection_drop_p=projection_drop_p,
                    qkv_bias=qkv_bias,
                ),
            )
        )
        self.mlp = ResidualAddition(
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                ViTMLPBlock(
                    embed_dim=embed_dim,
                    expansion=forward_expansion,
                    drop_p=forward_drop_p,
                    activation=activation,
                ),
            )
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.mlp(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop_p: float = 0.0,
        projection_drop_p: float = 0.2,
        qkv_bias: bool = False,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.2,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            ViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop_p=attn_drop_p,
                projection_drop_p=projection_drop_p,
                qkv_bias=qkv_bias,
                forward_expansion=forward_expansion,
                forward_drop_p=forward_drop_p,
                activation=activation,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        features = []
        for layer in self.layers:
            features.append(x)
            x = layer(x)
        features.append(self.norm(x))
        return features


class ViTBackbone(Backbone):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop_p: float = 0.0,
        projection_drop_p: float = 0.2,
        qkv_bias: bool = False,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.2,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.embedder = ViTPatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
        )

        self.encoder = ViTEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop_p=attn_drop_p,
            projection_drop_p=projection_drop_p,
            qkv_bias=qkv_bias,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p,
            activation=activation,
        )

    def forward(self, pixel_values: Tensor) -> List[Tensor]:
        embeddings = self.embedder(pixel_values)
        features = self.encoder(embeddings)
        return features
