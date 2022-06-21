from torch import nn
from glasses.config import ConfigMixin

from glasses.models.vision.classification.outputs import ModelForClassificationOutput
from .base import HeadForClassification
from typing import List
from torch import Tensor

from dataclasses import dataclass


@dataclass
class StupidHeadConfig:
    foo: str


class StupidHead(HeadForClassification, ConfigMixin):
    def __init__(self, foo: str):
        super().__init__()

    def forward(self, features):
        x = features[-1]
        return x
