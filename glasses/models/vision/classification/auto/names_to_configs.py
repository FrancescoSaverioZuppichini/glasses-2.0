from ..dummy.config import DummyForClassificationConfig, DummyConfig, LinearHeadConfig

NAMES_TO_CONFIGS = {
    "dummy-d0": DummyForClassificationConfig(DummyConfig(), LinearHeadConfig(10, 64)),
    "dummy-d1": DummyForClassificationConfig(
        DummyConfig(in_channels=128), LinearHeadConfig(10, 128)
    ),
}
