from dataclasses import dataclass
from glasses.config import Config


@dataclass
class AnyModelForClassificationConfig:
    backbone_config: Config
    head_config: Config
