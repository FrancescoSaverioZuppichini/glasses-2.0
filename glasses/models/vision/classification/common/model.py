from __future__ import annotations
from torch import nn
from ..outputs import ModelForClassificationOutput
from glasses.config import Config
from glasses.models.vision.backbones.auto.configs_to_models import (
    CONFIGS_TO_MODELS as BACKBONE_CONFIGS_TO_MODELS,
)
from ..heads.auto.configs_to_models import CONFIGS_TO_MODELS as HEADS_CONFIGS_TO_MODELS
from torch import Tensor
from .config import AnyModelForClassificationConfig
from typing import List

from ..base import ModelForClassification


class AnyModelForClassification(ModelForClassification):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values: Tensor) -> ModelForClassificationOutput:
        features: List[Tensor] = self.backbone(pixel_values)
        logits: Tensor = self.head(features)
        return {"logits": logits}

    @classmethod
    def from_config(
        cls, config: AnyModelForClassificationConfig
    ) -> AnyModelForClassification:
        backbone_config: Config = config.backbone_config
        head_config: Config = config.head_config

        backbone_func = BACKBONE_CONFIGS_TO_MODELS[type(backbone_config)]
        head_func = HEADS_CONFIGS_TO_MODELS[type(head_config)]

        return cls(
            backbone_func.from_config(backbone_config),
            head_func.from_config(head_config),
        )
