from typing import Any, List
from torch import nn, Tensor
from ..outputs import ModelOutputForClassification


class HeadForClassification(nn.Module):
    def forward(self, features: List[Tensor]) -> Tensor:
        raise NotImplemented
