from typing import TypedDict
from torch import Tensor

# [TODO] should be ModelForClassificationOutput
class ModelOutputForClassification(TypedDict):
    logits: Tensor
