from glasses.models.auto import AutoModel
from glasses.models.auto.utils import (
    get_configs_to_models_map,
    get_names_to_configs_map,
)


class AutoModelBackbone(AutoModel):
    names_to_configs = get_names_to_configs_map("glasses.models.vision.backbones")
    configs_to_models = get_configs_to_models_map("glasses.models.vision.backbones")


class AutoModelForClassification(AutoModel):
    names_to_configs = get_names_to_configs_map("glasses.models.vision.classification")
    configs_to_model = get_configs_to_models_map("glasses.models.vision.classification")
