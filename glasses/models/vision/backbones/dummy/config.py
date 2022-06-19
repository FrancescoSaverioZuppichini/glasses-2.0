from glasses.Config import Config
from dataclasses import dataclass


@dataclass
class DummyConfig(Config):
    in_channels: int = 3
    out_channels: int = 64
