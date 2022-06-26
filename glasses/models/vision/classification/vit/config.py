from dataclasses import dataclass

from glasses.models.vision.backbones.vit.config import ViTBackboneConfig
from glasses.models.vision.classification.heads.linear_head import \
    LinearHeadConfig

from ..common import AnyModelForClassificationConfig
from ..heads.vit import ViTHeadConfig


@dataclass
class ViTForClassificationConfig(AnyModelForClassificationConfig):
    """Config for [`ViT`](/models/vision/classification/vit) model"""

    backbone_config: ViTBackboneConfig
    head_config: ViTHeadConfig
