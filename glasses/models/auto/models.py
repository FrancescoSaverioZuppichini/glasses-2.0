from typing import Optional

from .config import AutoConfig
from .base import Auto
from glasses.config import Config
from torch import nn

NAMES_TO_MODEL = {
    # here we add our models
}


class AutoModel(Auto):
    """

    Usage:

    ```python

    my_model: Any = AutoModel.from_name("my_model")

    ```

    """

    @staticmethod
    def from_name(name: str, config: Optional[Config]) -> nn.Module:
        config: Config = AutoConfig.from_name(name) if not config else config
        model = None
        # get config
        # get model type

    # def from_pretrained()
