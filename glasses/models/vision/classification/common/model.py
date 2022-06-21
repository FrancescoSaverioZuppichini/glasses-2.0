from dataclasses import dataclass
from torch import nn
from ..outputs import ModelForClassificationOutput
from glasses.config import ConfigMixin, Config
from glasses.models.vision.backbones.config_map import (
    CONFIGS_TO_MODELS as BACKBONE_CONFIGS_TO_MODELS,
)
from ..heads.config_map import CONFIGS_TO_MODELS as HEADS_CONFIGS_TO_MODELS

from .config import AnyModelForClassificationConfig


class AnyModelForClassification(nn.Module, ConfigMixin):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values) -> ModelForClassificationOutput:
        features = self.backbone(pixel_values)
        logits = self.head(features)
        return {"logits": logits}

    @classmethod
    def from_config(cls, config: AnyModelForClassificationConfig):
        backbone_config: Config = config.backbone_config
        head_config: Config = config.head_config

        backbone_func = BACKBONE_CONFIGS_TO_MODELS[type(backbone_config)]
        head_func = HEADS_CONFIGS_TO_MODELS[type(head_config)]

        return cls(
            backbone_func.from_config(backbone_config),
            head_func.from_config(head_config),
        )
