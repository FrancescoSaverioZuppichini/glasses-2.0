from dataclasses import dataclass

from glasses.config import Config

from .model import AnyModelForClassification


@dataclass
class AnyModelForClassificationConfig(Config):
    backbone_config: Config
    head_config: Config

    def build(self) -> AnyModelForClassification:
        backbone = self.backbone_config.build()
        head = self.head_config.build()
        return AnyModelForClassification(backbone, head)
