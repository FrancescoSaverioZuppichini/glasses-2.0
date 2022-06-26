from dataclasses import dataclass
from typing import Tuple

from torch import nn

from glasses.config import Config

from .model import LeViTBackbone


@dataclass
class LeViTBackboneConfig(Config):
    img_size: int = 224
    in_channels: int = 3
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    patch_size: int = 16
    hidden_sizes: Tuple[int] = (128, 256, 384)
    num_heads: Tuple[int] = (4, 8, 12)
    depths: Tuple[int] = (4, 4, 4)
    key_dim: Tuple[int] = (16, 16, 16)
    drop_path_rate: float = 0.0
    mlp_ratio: Tuple[int] = (2, 2, 2)
    attn_ratio: Tuple[int] = (2, 2, 2)

    def build(self) -> LeViTBackbone:
        return LeViTBackbone(**self.__dict__)
