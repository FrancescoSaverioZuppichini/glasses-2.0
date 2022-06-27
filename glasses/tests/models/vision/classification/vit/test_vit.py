from glasses.models.vision.classification.vit.config import (
    ViTForClassificationConfig,
    ViTHeadConfig,
)

from glasses.models.vision.backbones.vit import ViTBackboneConfig
from glasses.tests.model_tester import model_tester
from glasses.models.vision.classification import ModelForClassificationOutput
import torch


def get_test_config(policy: str):
    return ViTForClassificationConfig(
        backbone_config=ViTBackboneConfig(
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
    configs = [get_test_config(policy="token"), get_test_config(policy="mean")]
    for config in configs:
        input_dict = {
            "pixel_values": torch.randn(
                (
                    2,
                    config.backbone_config.in_channels,
                    config.backbone_config.img_size,
                    config.backbone_config.img_size,
                )
            )
        }
        output_shape_dict = {"logits": (2, config.head_config.num_classes)}
        model_tester(
            config, input_dict, output_shape_dict, ModelForClassificationOutput
        )
