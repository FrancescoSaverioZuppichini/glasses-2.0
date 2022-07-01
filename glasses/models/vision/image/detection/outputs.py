from typing import TypedDict

from torch import Tensor


class ModelForImageDetectionOutput(TypedDict):
    """The output for image detection models."""

    logits: Tensor
    """A `torch.Tensor` of shape `(batch_size, num_bboxes, num_classes + 1)`."""
    bboxes: Tensor
    """A `torch.Tensor` of shape `(batch_size, num_bboxes, 4)`."""
