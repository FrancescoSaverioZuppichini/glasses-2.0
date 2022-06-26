from __future__ import annotations

from typing import List

from torch import Tensor, nn

from ..base import ModelForClassification
from ..outputs import ModelForClassificationOutput


class AnyModelForClassification(ModelForClassification):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values: Tensor) -> ModelForClassificationOutput:
        features: List[Tensor] = self.backbone(pixel_values)
        logits: Tensor = self.head(features)
        return {"logits": logits}
