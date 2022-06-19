from typing import TypedDict
from torch import Tensor


class ModelOutputForClassification(TypedDict):
    logits: Tensor
