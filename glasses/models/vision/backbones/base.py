from typing import List
from torch import nn, Tensor


class Backbone(nn.Module):
    def forward(self, pixel_values: Tensor) -> List[Tensor]:
        raise NotImplemented
