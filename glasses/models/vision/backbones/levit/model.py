import itertools
from typing import Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..base import Backbone
from ..utils import DropPath


class LeViTConvEmbeddings(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convolution(x)
        x = self.bn(x)
        return x

class LeViTPatchEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: Tuple[int],
        kernel_size: int, 
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.patch_embed = nn.Sequential(
            LeViTConvEmbeddings(in_channels, hidden_size[0] // 8, kernel_size, stride, padding),
            nn.Hardswish(),
            LeViTConvEmbeddings(hidden_size[0] // 8, hidden_size[0] // 4, kernel_size, stride, padding),
            nn.Hardswish(),
            LeViTConvEmbeddings(hidden_size[0] // 4, hidden_size[0] // 2, kernel_size, stride, padding),
            nn.Hardswish(),
            LeViTConvEmbeddings(hidden_size[0] // 2, hidden_size[0], kernel_size, stride, padding),
        )
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        in_channels = x.shape[1]
        if in_channels != self.in_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class LeViTMLPLayerWithBN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.bn(x.flatten(0, 1)).reshape_as(x)
        return x

class LeViTSubsample(nn.Module):
    def __init__(self, stride: int, resolution: int):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, channels = x.shape
        x = x.view(batch_size, self.resolution, self.resolution, channels)[
            :, :: self.stride, :: self.stride
        ].reshape(batch_size, -1, channels)
        return x

class LeViTAttentionBlock(nn.Module):
    def __init__(self, hidden_sizes: Tuple[int], key_dim: Tuple[int], num_heads: Tuple[int], attn_ratio: Tuple[int], resolution: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attn_ratio = attn_ratio
        self.out_dim_kv = attn_ratio * key_dim * num_heads + key_dim * num_heads * 2
        self.out_dim_projection = attn_ratio * key_dim * num_heads

        self.qkv = LeViTMLPLayerWithBN(hidden_sizes, self.out_dim_kv)
        self.act = nn.Hardswish()
        self.projection = LeViTMLPLayerWithBN(self.out_dim_projection, hidden_sizes)

        points = list(itertools.product(range(resolution), range(resolution)))
        len_points = len(points)
        attention_offsets, indices = {}, []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        self.attention_bias_cache = {}
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(indices).view(len_points, len_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.view(batch_size, seq_length, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.attn_ratio * self.key_dim], dim=3
        )
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(x.device)
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch_size, seq_length, self.out_dim_projection)
        x = self.projection(self.act(x))
        return x

class LeViTAttentionSubsampleBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key_dim: Tuple[int],
        num_heads: Tuple[int],
        attn_ratio: Tuple[int],
        stride,
        resolution_in: int,
        resolution_out: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attn_ratio = attn_ratio
        self.out_dim_kv = attn_ratio * key_dim * num_heads + key_dim * num_heads
        self.out_dim_projection = attn_ratio * key_dim * num_heads
        self.resolution_out = resolution_out
        # resolution_in is the intial resolution, resoloution_out is final resolution after downsampling
        self.kv = LeViTMLPLayerWithBN(input_dim, self.out_dim_kv)
        self.queries_subsample = LeViTSubsample(stride, resolution_in)
        self.queries = LeViTMLPLayerWithBN(input_dim, key_dim * num_heads)
        self.act = nn.Hardswish()
        self.projection = LeViTMLPLayerWithBN(self.out_dim_projection, output_dim)

        self.attention_bias_cache = {}

        points = list(itertools.product(range(resolution_in), range(resolution_in)))
        points_ = list(itertools.product(range(resolution_out), range(resolution_out)))
        len_points, len_points_ = len(points), len(points_)
        attention_offsets, indices = {}, []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(indices).view(len_points_, len_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape
        key, value = (
            self.kv(x)
            .view(batch_size, seq_length, self.num_heads, -1)
            .split([self.key_dim, self.attn_ratio * self.key_dim], dim=3)
        )
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        query = self.queries(self.queries_subsample(x))
        query = query.view(batch_size, self.resolution_out**2, self.num_heads, self.key_dim).permute(
            0, 2, 1, 3
        )

        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(x.device)
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch_size, -1, self.out_dim_projection)
        x = self.projection(self.act(x))
        return x

class LeViTMLPBLock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = LeViTMLPLayerWithBN(input_dim, hidden_dim)
        self.act = nn.Hardswish()
        self.fc2 = LeViTMLPLayerWithBN(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ResidualAddition(nn.Module):
    def __init__(self, module: nn.Module, drop_rate: float):
        super().__init__()
        self.module = module
        self.drop_rate = drop_rate

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.drop_rate > 0:
            rnd = torch.rand(x.size(0), 1, 1, device=x.device)
            rnd = rnd.ge_(self.drop_rate).div(1 - self.drop_rate).detach()
            x = x + self.module(x) * rnd
            return x
        else:
            x = x + self.module(x)
            return x

class LeViTBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        down_ops,
        resolution_in: int,
        hidden_sizes: int,
        num_heads: int,
        depths: int,
        key_dim: int,
        drop_path_rate: float,
        mlp_ratio: int,
        attn_ratio: int,
        hidden_sizes_tuple: Tuple[int],
    ):
        super().__init__()
        self.layers = []
        self.resolution_in = resolution_in
        # resolution_in is the intial resolution, resolution_out is final resolution after downsampling
        for _ in range(depths):
            self.layers.append(
                ResidualAddition(
                    LeViTAttentionBlock(hidden_sizes, key_dim, num_heads, attn_ratio, resolution_in),
                    drop_path_rate,
                )
            )
            if mlp_ratio > 0:
                hidden_dim = hidden_sizes * mlp_ratio
                self.layers.append(
                    ResidualAddition(LeViTMLPBLock(hidden_sizes, hidden_dim), drop_path_rate)
                )

        if down_ops[0] == "Subsample":
            self.resolution_out = (self.resolution_in - 1) // down_ops[5] + 1
            print(hidden_sizes)
            self.layers.append(
                LeViTAttentionSubsampleBlock(
                    *hidden_sizes_tuple[idx : idx + 2],
                    key_dim=down_ops[1],
                    num_heads=down_ops[2],
                    attn_ratio=down_ops[3],
                    stride=down_ops[5],
                    resolution_in=resolution_in,
                    resolution_out=self.resolution_out,
                )
            )
            self.resolution_in = self.resolution_out
            if down_ops[4] > 0:
                hidden_dim = hidden_sizes_tuple[idx + 1] * down_ops[4]
                self.layers.append(
                    ResidualAddition(
                        LeViTMLPBLock(hidden_sizes_tuple[idx + 1], hidden_dim), drop_path_rate
                    )
                )

        self.layers = nn.ModuleList(self.layers)

    def get_resolution(self):
        return self.resolution_in

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LeViTEncoder(nn.Module):
    """
    LeViT Encoder consisting of multiple `LevitStage` stages.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        hidden_sizes: Tuple[int] = (128, 256, 384),
        num_heads: Tuple[int] = (4, 8, 12),
        depths: Tuple[int] = (4, 4, 4),
        key_dim: Tuple[int] = (16, 16, 16),
        drop_path_rate: float = 0.0,
        mlp_ratio: Tuple[int] = (2, 2, 2),
        attn_ratio: Tuple[int] = (2, 2, 2),
    ):
        super().__init__()
        down_ops = [
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]
        resolution = img_size // patch_size
        self.stages = []
        down_ops.append([""])

        for stage_idx in range(len(depths)):
            stage = LeViTBlock(
                stage_idx,
                down_ops[stage_idx],
                resolution,
                hidden_sizes[stage_idx],
                key_dim[stage_idx],
                depths[stage_idx],
                num_heads[stage_idx],
                attn_ratio[stage_idx],
                mlp_ratio[stage_idx],
                drop_path_rate,
                hidden_sizes_tuple=hidden_sizes
            )
            resolution = stage.get_resolution()
            self.stages.append(stage)

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

class LeViTBackbone(Backbone):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        patch_size: int = 16,
        hidden_sizes: Tuple[int] = (128, 256, 384),
        num_heads: Tuple[int] = (4, 8, 12),
        depths: Tuple[int] = (4, 4, 4),
        key_dim: Tuple[int] = (16, 16, 16),
        drop_path_rate: float = 0.0,
        mlp_ratio: Tuple[int] = (2, 2, 2),
        attn_ratio: Tuple[int] = (2, 2, 2),
    ):
        super().__init__()
        self.embedder = LeViTPatchEmbeddings(
            in_channels=in_channels,
            hidden_size=hidden_sizes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.encoder = LeViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            depths=depths,
            hidden_sizes=hidden_sizes,
            num_heads=num_heads,
            key_dim=key_dim,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
            attn_ratio=attn_ratio
        )

    def forward(self, pixel_values: Tensor) -> List[Tensor]:
        embeddings = self.embedder(pixel_values)
        features = self.encoder(embeddings)
        return features
