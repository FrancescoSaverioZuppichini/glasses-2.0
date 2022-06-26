from dataclasses import dataclass

from glasses.config import Config

from .model import Dummy


@dataclass
class DummyConfig(Config):
    in_channels: int = 3
    out_channels: int = 64

    def build(self):
        return Dummy(**self.__dict__)
