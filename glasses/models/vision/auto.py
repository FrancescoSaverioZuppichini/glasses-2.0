from glasses.models.auto import AutoModel
from glasses.models.auto.utils import get_names_to_configs_map


class AutoModelBackbone(AutoModel):
    names_to_configs = get_names_to_configs_map("glasses.models.vision.backbones")


class AutoModelForImageClassification(AutoModel):
    names_to_configs = get_names_to_configs_map(
        "glasses.models.vision.image.classification"
    )
