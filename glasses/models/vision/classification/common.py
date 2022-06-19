from dataclasses import dataclass
from torch import nn
from glasses.Config import Config
from .outputs import ModelOutputForClassification
from glasses.Configurable import Configurable
from glasses.CONFIG_MAP import CONFIG_MAP

class AnyModelForSegmentation(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs


@dataclass
class AnyModelForClassificationConfig:
    backbone_config: Config
    head_config: Config


class AnyModelForClassification(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values) -> ModelOutputForClassification:
        features = self.backbone(pixel_values)
        logits = self.head(features)
        return {"logits": logits}

    @classmethod
    def from_config(cls, config: AnyModelForClassificationConfig):
        backbone_config: Config = config.backbone_config
        head_config: Config = config.head_config

        backbone_func: Configurable = CONFIG_MAP[type(backbone_config)]
        head_func: Configurable = CONFIG_MAP[type(head_config)]

        return cls(
            backbone_func.from_config(backbone_config),
            head_func.from_config(head_config),
        )
