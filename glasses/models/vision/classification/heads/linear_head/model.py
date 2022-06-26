from typing import List

from torch import Tensor, nn

from ..base import HeadForClassification


class LinearHead(HeadForClassification):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
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
