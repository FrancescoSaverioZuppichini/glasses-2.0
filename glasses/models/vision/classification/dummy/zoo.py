from glasses.models.auto import ModelZoo
from glasses.models.vision.backbones.dummy.config import DummyConfig
from glasses.models.vision.classification.heads.linear_head.config import (
    LinearHeadConfig,
)

from .config import DummyForClassificationConfig


def dummy_d0_im():
    return DummyForClassificationConfig(
        backbone_config=DummyConfig(3, 64), head_config=LinearHeadConfig(64, 1000)
    )


def dummy_d1_im():
    return DummyForClassificationConfig(
        backbone_config=DummyConfig(3, 128), head_config=LinearHeadConfig(128, 1000)
    )


zoo = ModelZoo(dummy_d0_im=dummy_d0_im, dummy_d1_im=dummy_d1_im)
