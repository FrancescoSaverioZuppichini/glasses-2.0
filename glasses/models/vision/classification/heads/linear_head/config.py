from dataclasses import dataclass
from glasses.config import Config


@dataclass
class LinearHeadConfig(Config):
    in_channels: int
    num_classes: int
