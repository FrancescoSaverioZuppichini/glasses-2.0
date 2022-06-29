from dataclasses import dataclass

from glasses.models.vision.backbones.vit.config import ViTBackboneConfig

from ..common import AnyModelForImageClassificationConfig
from ..heads.vit import ViTHeadConfig


@dataclass
class ViTForImageClassificationConfig(AnyModelForImageClassificationConfig):
    """Config for [`ViT`](/models/vision/image/classification/vit) model"""

    backbone_config: ViTBackboneConfig
    head_config: ViTHeadConfig
