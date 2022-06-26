from glasses.models import ModelZoo

from .config import ViTBackboneConfig


def vit_small_patch16_224():
    return ViTBackboneConfig(depth=8, num_heads=8, forward_expansion=3)


def vit_base_patch16_224():
    return ViTBackboneConfig(depth=12, num_heads=12, forward_expansion=4, qkv_bias=True)


def vit_base_patch16_384():
    return ViTBackboneConfig(
        img_size=384, depth=12, num_heads=12, forward_expansion=4, qkv_bias=True
    )


zoo = ModelZoo(
    vit_small_patch16_224=vit_small_patch16_224,
    vit_base_patch16_224=vit_base_patch16_224,
    vit_base_patch16_384=vit_base_patch16_384,
)
