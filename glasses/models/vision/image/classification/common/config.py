from dataclasses import dataclass

from glasses.config import Config

from .model import AnyModelForImageClassification


@dataclass
class AnyModelForImageClassificationConfig(Config):
    backbone_config: Config
    head_config: Config

    def build(self) -> AnyModelForImageClassification:
        backbone = self.backbone_config.build()
        head = self.head_config.build()
        return AnyModelForImageClassification(backbone, head)
