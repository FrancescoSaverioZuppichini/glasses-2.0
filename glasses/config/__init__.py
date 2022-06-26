from __future__ import annotations
from dataclasses import dataclass
from torch import nn


@dataclass
class Config:
    """
    Base class for configurations, all configuration **must** inherit from this class.

    !!! note
        A `Config` holds what is needed to create the inputs to your class. It's purely a **data container**. Thus, it **must not have any side effect**, any logic that requires the `Config` values to be somehow processed must be implemented somewhere else. A correct approach is to override the [`ConfigMixin.from_config`](#glasses.config.ConfigMixin.from_config) method.

    A custom configuration can be written as follows:

    ```python
    from glasses.config import Config

    @dataclass
    class MyConfig(Config):
        foo: int
        baa: str
    ```
    """

    pass

    def build(self) -> nn.Module:
        raise NotImplemented


# class ConfigMixin:
#     """
#     A mixin that add a `from_config` **classmethod** to your class.

#     Override this method to create your class' inputs based on the `Config`.

#     By default, the passed configuration to `from_config` is converted to dict.

#     """

#     @classmethod
#     def from_config(cls, config: ConfigMixin._config_type) -> ConfigMixin:
#         """A class method that takes a `Config` an return the class correctly initialized using the `config` values.

#         Args:
#             config (Config): A `Config` instance.

#         Returns:
#             ConfigMixin: The class instance.
#         """
#         return cls(**config.__dict__)
