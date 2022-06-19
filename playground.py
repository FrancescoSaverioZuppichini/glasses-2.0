from glasses.CONFIG_MAP import CONFIG_MAP


from glasses.models.vision.classification.heads.LinearHead import (
    LinearHeadConfig,
    LinearHead,
)

from glasses.models.vision.backbones.dummy.config import DummyConfig
from glasses.models.vision.backbones.dummy.model import Dummy
from glasses.models.vision.classification.common import (
    AnyModelForClassification,
    AnyModelForClassificationConfig,
    Configurable,
)

import torch
from glasses.models.vision.classification.heads.stupid import StupidHeadConfig

from glasses.models.vision.classification.outputs import ModelOutputForClassification

my_model = AnyModelForClassification(backbone=Dummy(3, 64), head=LinearHead(10, 64))

x = torch.randn((1, 3, 224, 224))
out: ModelOutputForClassification = my_model(x)
print(out["logits"].shape)

from dataclasses import dataclass


@dataclass
class DummyForClassification(AnyModelForClassificationConfig):
    backbone_config: DummyConfig
    head_config: LinearHeadConfig


zoo = {
    "dummy-for-classification": DummyForClassification(
        DummyConfig(1, 128), LinearHeadConfig(20, 128)
    )
}


config = zoo["dummy-for-classification"]

config.head_config = LinearHeadConfig(20, 128)


# config.backbone_config.in_channels

my_model = AnyModelForClassification.from_config(config)
print(my_model)
# x = torch.randn((1, 1, 224, 224))
# out: ModelOutputForClassification = my_model(x)
# print(out["logits"].shape)
