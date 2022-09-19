import torch

from glasses.models.vision.backbones.vit import ViTBackboneConfig
from glasses.models.vision.image.classification import \
    ModelForImageClassificationOutput
from glasses.models.vision.image.classification.vit.config import (
    ViTForImageClassificationConfig, ViTHeadConfig)
from glasses.utils.model_tester import model_tester


def get_test_config(policy: str) -> ViTForImageClassificationConfig:
    return ViTForImageClassificationConfig(
        backbone_config=ViTBackboneConfig(
            # let's keep it small
            img_size=24,
            patch_size=8,
            depth=3,
            embed_dim=16,
            num_heads=2,
            forward_expansion=2,
        ),
        head_config=ViTHeadConfig(emb_size=16, num_classes=10, policy=policy),
    )


def test_vit():
    batch_size = 2
    # create a couple of configs
    configs = [get_test_config(policy="token"), get_test_config(policy="mean")]
    for config in configs:
        # create the input dict, something that 
        input_dict = {
            "pixel_values": torch.randn(
                (
                    batch_size,
                    config.backbone_config.in_channels,
                    config.backbone_config.img_size,
                    config.backbone_config.img_size,
                )
            )
        }
        output_shape_dict = {"logits": (batch_size, config.head_config.num_classes)}
        model_tester(
            config, input_dict, output_shape_dict, ModelForImageClassificationOutput
        )
