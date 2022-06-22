from .base import Auto
from glasses.config import Config
from torch import nn
from typing import Callable

CONFIG_TO_MODELS_TYPE = {
    # here we add our configurations
}


class AutoConfig(Auto):
    """

    Usage:

    ```python

    my_config: Config = AutoConfig.from_name("my_config")

    ```    _config_type: Config = Config


    """

    @staticmethod
    def get_model_type(config: Config) -> Callable[[], nn.Module]:
        # get the config type
        model_type = None
        return model_type
