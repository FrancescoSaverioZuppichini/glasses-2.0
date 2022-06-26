from dataclasses import dataclass

from glasses.config import Config

from .model import LeViTHead


@dataclass
class LeViTHeadConfig(Config):
    hidden_size: int = 384
    num_classes: int = 1000

    def build(self) -> LeViTHead:
        return LeViTHead(**self.__dict__)
