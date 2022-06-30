from typing import Any, List

from torch import Tensor, nn

from .outputs import ModelForImageClassificationOutput


class ModelForImageClassification(nn.Module):
    """Base class for classification models

    Define a custom image classification model. It can be anything you want, the only contrain is that it **must** return a `ModelForImageClassificationOutput`.

    ```python

    class MyModelForImageClassification(ModelForImageClassification):
        def __init__(self, in_channels: int, num_classes: int ):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, 64, kernel_size=3)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flat = nn.Flatten()
            self.fc = nn.Linear(64, num_classes)

        def forward(self, pixel_values: Tensor) -> ModelForImageClassificationOutput:
            x = self.conv(pixel_values)
            x = self.pool(x)
            x = self.flat(x)
            x = self.fc(x)
            return {"logits": x}

    ```

    The above example is a fixed models, it doesn't have composable part. In reality, classification models are (usually) composed by a **backbone** and a **head**. Since all the backbones and heads in glasses must follow known rules, it trivial to compose them.


    ```python

    from glasses.models.vision.backbones import ResNet

    class ResNetForImageClassification(ModelForImageClassification):
        def __init__(self, in_channels: int, ..., num_classes: int):
            super().__init__()
            self.backbone =ResNet(in_channels, ....)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flat = nn.Flatten()
            self.fc = nn.Linear(64, num_classes)

        def forward(self, pixel_values: Tensor) -> ModelForImageClassificationOutput:
            features = self.backbone(pixel_values)
            x = features[-1]
            x = self.pool(x)
            x = self.flat(x)
            x = self.fc(x)
            return {"logits": x}

    ```

    In 99% of cases you will take advantage of the [`AnyModelForImageClassification`]() that allows you to mix on the fly any backbone and classification head in glasses.

    """

    def forward(self, pixel_values: Tensor) -> ModelForImageClassificationOutput:
        """The forward method for classification head.

        Args:
            pixel_values (Tensor):  The input image.

        Returns:
            Tensor: The logits.
        """
        raise NotImplemented
