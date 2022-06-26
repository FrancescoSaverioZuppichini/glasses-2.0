from __future__ import annotations
from torch import nn

from ..outputs import ModelForClassificationOutput
from torch import Tensor
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
