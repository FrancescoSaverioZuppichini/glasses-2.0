from typing import Any, List

from torch import Tensor, nn

from ..base import Backbone


class Dummy(Backbone):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(self, pixel_values: Tensor) -> List[Tensor]:
        return [self.conv(pixel_values)]
