from typing import List

from torch import Tensor, nn

from ..base import HeadForClassification


class LeViTHead(HeadForClassification):
    def __init__(self, hidden_size: int = 384, num_classes: int = 1000):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, features: List[Tensor]) -> Tensor:
        x = features.mean(1)
        x = self.fc(self.bn(x))
        return x
