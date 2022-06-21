from .dummy.config import DummyForClassificationConfig, DummyConfig, LinearHeadConfig

from .config_map import CONFIGS_TO_MODELS

NAMES_TO_CONFIG = {
    "dummy-d0": DummyForClassificationConfig(DummyConfig(), LinearHeadConfig(10, 64)),
    "dummy-d1": DummyForClassificationConfig(
        DummyConfig(in_channels=128), LinearHeadConfig(10, 128)
    ),
}


class AutoModelForClassification:
    @staticmethod
    def from_name(name: str):
        cfg = NAMES_TO_CONFIG[name]
        model_func = CONFIGS_TO_MODELS[type(cfg)]
        return model_func.from_config(cfg)
