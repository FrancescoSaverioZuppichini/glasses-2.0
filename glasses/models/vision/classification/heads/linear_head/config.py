from dataclasses import dataclass
from glasses.config import Config
from .model import LinearHead

@dataclass
class LinearHeadConfig(Config):
    in_channels: int
    num_classes: int

    def build(self):
        return LinearHead(**self.__dict__)