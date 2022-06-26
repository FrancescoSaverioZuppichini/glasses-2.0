from glasses.models.auto import ModelZoo
from glasses.models.vision.backbones.dummy.config import DummyConfig
from glasses.models.vision.classification.heads.linear_head.config import (
    LinearHeadConfig,
)
from .config import DummyForClassificationConfig

zoo = ModelZoo(
    dummy_d0_im=DummyForClassificationConfig(
        backbone_config=DummyConfig(3, 64), head_config=LinearHeadConfig(64, 1000)
    )
)
