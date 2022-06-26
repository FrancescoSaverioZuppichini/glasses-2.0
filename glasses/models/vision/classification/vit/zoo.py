from glasses.models.auto import ModelZoo
from glasses.models.vision.backbones.vit.config import ViTBackboneConfig

from ..heads.vit import ViTHeadConfig
from .config import ViTForClassificationConfig


def vit_small_patch16_224():
    return ViTForClassificationConfig(
        backbone_config=ViTBackboneConfig(depth=8, num_heads=8, forward_expansion=3),
        head_config=ViTHeadConfig(768, 1000),
    )


def vit_base_patch16_224():
    return ViTForClassificationConfig(
        backbone_config=ViTBackboneConfig(
            depth=12, num_heads=12, forward_expansion=4, qkv_bias=True
        ),
        head_config=ViTHeadConfig(768, 1000),
    )


def vit_base_patch16_384():
    return ViTForClassificationConfig(
        backbone_config=ViTBackboneConfig(
            img_size=384, depth=12, num_heads=12, forward_expansion=4, qkv_bias=True
        ),
        head_config=ViTHeadConfig(768, 1000),
    )


zoo = ModelZoo(
    vit_small_patch16_224=vit_small_patch16_224,
    vit_base_patch16_224=vit_base_patch16_224,
    vit_base_patch16_384=vit_base_patch16_384,
)
