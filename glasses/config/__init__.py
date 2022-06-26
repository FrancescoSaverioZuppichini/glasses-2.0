from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass
class Config:
    """
    Base class for configurations, all configuration **must** inherit from this class. For a in depth tutorial about configs, head over [Configurations](/)

    !!! important
        Models are not coupled with `Config`, therefore they are unaware of the configuration system. Each `Config` is linked to a specific model, not viceversa.

    !!! note
        A `Config` holds what is needed to create a model. Therefore, they are perfect to share custom version of a specific architecture. A `Config` is **data container**. Thus, it **must not have any side effect**, any logic that requires the `Config` values to be somehow processed must be implemented in the [`Config.build`](#glasses.config.Config.build) function.


    A custom configuration can be written as follows:

    ```python
    from glasses.config import Config

    # Assume we have a model
    class MyModel(nn.Module):
        def __init__(in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    # Let's create it's configuration
    @dataclass
    class MyConfig(Config):
        in_channels: int
        out_channels: int

        def build(self) -> nn.Module:
            # create a `MyModel` instance using `MyConfig`
            return MyModel(**self.__dict__)

    model: MyModel = MyConfig(2, 2).build()
    ```

    Each Config is linked to a specific model, not viceversa. Models had no idea about the configuration system and can be created normally as you expect with their constructor.

    ## Nested Configurations

    """

    def build(self) -> nn.Module:
        raise NotImplemented
