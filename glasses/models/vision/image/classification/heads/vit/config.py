from dataclasses import dataclass

from glasses.config import Config

from .model import ViTHead


@dataclass
class ViTHeadConfig(Config):
    emb_size: int = 768
    num_classes: int = 1000
    policy: str = "token"

    def build(self) -> ViTHead:
        return ViTHead(**self.__dict__)
