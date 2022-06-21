from enum import auto
from glasses.models.vision.classification import AutoModelForClassification
from glasses.models.vision.classification.dummy.config import (
    DummyForClassificationConfig,
)
from glasses.storage import LocalStorage
from dataclasses import asdict
from glasses.models.vision.classification.heads.stupid import StupidHeadConfig

name = "dummy-d0"
auto_model = AutoModelForClassification()

# model, config = auto_model.from_name(name)


# auto_model.storage.put(name, model.state_dict(), asdict(config))
config: DummyForClassificationConfig = auto_model.names_to_configs[name]
config.head_config = StupidHeadConfig(foo="str")
model, config = auto_model.from_pretrained(name, config)
print(model)
