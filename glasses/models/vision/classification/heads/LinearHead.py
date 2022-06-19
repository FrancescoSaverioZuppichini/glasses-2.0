from torch import nn
from glasses.Configurable import Configurable

from glasses.models.vision.classification.outputs import ModelOutputForClassification
from .HeadForClassification import HeadForClassification
from typing import List
from torch import Tensor

from dataclasses import dataclass


@dataclass
class LinearHeadConfig:
    num_classes: int
    in_channels: int


class LinearHead(HeadForClassification, Configurable):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: List[Tensor]) -> Tensor:
        x = features[-1]
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x
