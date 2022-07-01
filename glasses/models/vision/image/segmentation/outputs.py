from typing import TypedDict

from torch import Tensor


# of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
class ModelForImageSegmentationOutput(TypedDict):
    """The output for image segmentation models."""

    pixel_logits: Tensor
    """A `torch.Tensor` of shape `(batch_size, num_classes, height, width)`.
    
    !!! note
        The `height` and `width` are usually smaller than the original image.
    """
