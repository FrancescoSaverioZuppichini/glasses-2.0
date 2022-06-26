from dataclasses import dataclass

from torch import nn

from glasses.config import Config

from .model import ViTBackbone


@dataclass
class ViTBackboneConfig(Config):
    img_size: int = 224
    in_channels: int = 3
    patch_size: int = 16
    depth: int = 12
    embed_dim: int = 768
    num_heads: int = 12
    attn_drop_p: float = 0.0
    projection_drop_p: float = 0.2
    qkv_bias: bool = False
    forward_expansion: int = 4
    forward_drop_p: float = 0.2
    activation: nn.Module = nn.GELU

    def build(self) -> ViTBackbone:
        return ViTBackbone(**self.__dict__)
