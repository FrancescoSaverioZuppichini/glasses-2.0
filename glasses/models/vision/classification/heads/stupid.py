from torch import nn
from glasses.Configurable import Configurable

from glasses.models.vision.classification.outputs import ModelOutputForClassification
from .HeadForClassification import HeadForClassification
from typing import List
from torch import Tensor

from dataclasses import dataclass


@dataclass
class StupidHeadConfig:
    foo: str


class StupidHead(Configurable):
    def __init__(self, foo: str):
        super().__init__()

    def forward(self, features):
        x = features[-1]
        return x
