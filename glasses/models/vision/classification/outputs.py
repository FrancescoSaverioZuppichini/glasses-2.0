from typing import TypedDict
from torch import Tensor


class ModelForClassificationOutput(TypedDict):
    """The output for image classification models."""

    logits: Tensor
