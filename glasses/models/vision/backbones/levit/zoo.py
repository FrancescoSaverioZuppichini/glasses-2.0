from glasses.models import ModelZoo

from .config import LeViTBackboneConfig


def levit_128S():
    return LeViTBackboneConfig(
        hidden_sizes=[128, 256, 384],
        num_heads=[4, 6, 8],
        depths=[2, 3, 4],
        key_dim=[16, 16, 16],
        drop_path_rate=0,
    )

def levit_128():
    return LeViTBackboneConfig(
        hidden_sizes=[128, 256, 384],
        num_heads=[4, 8, 12],
        depths=[4, 4, 4],
        key_dim=[16, 16, 16],
        drop_path_rate=0,
    )

def levit_192():
    return LeViTBackboneConfig(
        hidden_sizes=[192, 288, 384],
        num_heads=[3, 5, 6],
        depths=[4, 4, 4],
        key_dim=[32, 32, 32],
        drop_path_rate=0,
    )

def levit_256():
    return LeViTBackboneConfig(
        hidden_sizes=[256, 384, 512],
        num_heads=[4, 6, 8],
        depths=[4, 4, 4],
        key_dim=[32, 32, 32],
        drop_path_rate=0,
    )

def levit_384():
    return LeViTBackboneConfig(
        hidden_sizes=[384, 512, 768],
        num_heads=[6, 9, 12],
        depths=[4, 4, 4],
        key_dim=[32, 32, 32],
        drop_path_rate=0,
    )


zoo = ModelZoo(
    levit_128S=levit_128S,
    levit_128=levit_128S,
    levit_192=levit_192,
    levit_256=levit_256,
    levit_384=levit_384,

)
