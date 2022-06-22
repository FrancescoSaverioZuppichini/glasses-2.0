from typing import Any, List
from torch import nn, Tensor
from ..outputs import ModelForClassificationOutput
from glasses.config import ConfigMixin


class HeadForClassification(nn.Module, ConfigMixin):
    """Base class for classification heads

    Define a custom classification head

    ```python

    class LinearHead(HeadForClassification):
        def __init__(self, num_classes: int, in_channels: int):
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
    ```

    """

    def forward(self, features: List[Tensor]) -> Tensor:
        """The forward method for classification head.

        Args:
            features (List[Tensor]): A list of features.

        Raises:
            NotImplemented:

        Returns:
            Tensor: The logits
        """
        raise NotImplemented
