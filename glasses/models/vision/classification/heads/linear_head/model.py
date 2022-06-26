from torch import Tensor, nn
from ..base import HeadForClassification

class LinearHead(HeadForClassification):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.fc(features)
        return logits
