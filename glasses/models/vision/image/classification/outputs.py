from typing import TypedDict

from torch import Tensor


class ModelForImageClassificationOutput(TypedDict):
    """The output for image classification models."""

    logits: Tensor
    # `(batch_size, config.num_labels)`)
