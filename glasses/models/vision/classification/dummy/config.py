from dataclasses import dataclass

from glasses.models.vision.backbones.dummy.config import DummyConfig
from glasses.models.vision.classification.heads.LinearHead import LinearHeadConfig
from ..common import AnyModelForClassificationConfig


@dataclass
class DummyForClassificationConfig(AnyModelForClassificationConfig):
    backbone_config: DummyConfig
    head_config: LinearHeadConfig
