from dataclasses import dataclass

from glasses.models.vision.backbones.levit.config import LeViTBackboneConfig
from ..common import AnyModelForClassificationConfig
from ..heads.levit import LeViTHeadConfig


@dataclass
class LeViTForClassificationConfig(AnyModelForClassificationConfig):
    """Config for [`LeViT`](/models/vision/classification/levit) model"""

    backbone_config: LeViTBackboneConfig
    head_config: LeViTHeadConfig
