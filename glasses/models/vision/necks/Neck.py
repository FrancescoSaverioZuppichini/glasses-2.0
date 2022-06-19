from typing import List
from torch import nn, Tensor


class Neck(nn.Module):
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        raise NotImplemented
