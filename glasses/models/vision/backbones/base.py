from typing import List

from torch import Tensor, nn


class Backbone(nn.Module):
    def forward(self, pixel_values: Tensor) -> List[Tensor]:
        raise NotImplementedError
