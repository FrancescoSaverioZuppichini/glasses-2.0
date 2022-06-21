from glasses.models.auto.base import AutoModel

from .configs_to_models import CONFIGS_TO_MODELS
from .names_to_configs import NAMES_TO_CONFIGS


class AutoModelForClassification(AutoModel):
    names_to_configs = NAMES_TO_CONFIGS
    configs_to_models = CONFIGS_TO_MODELS
