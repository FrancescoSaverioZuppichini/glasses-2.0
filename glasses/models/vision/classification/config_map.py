from .common import AnyModelForClassification, AnyModelForClassificationConfig
from .dummy.config import DummyForClassificationConfig

CONFIGS_TO_MODELS = {
    AnyModelForClassificationConfig: AnyModelForClassification,
    DummyForClassificationConfig: AnyModelForClassification,
}
