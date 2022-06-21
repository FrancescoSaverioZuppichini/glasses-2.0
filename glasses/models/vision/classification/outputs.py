from typing import TypedDict
from torch import Tensor


class ModelForClassificationOutput(TypedDict):
    logits: Tensor
