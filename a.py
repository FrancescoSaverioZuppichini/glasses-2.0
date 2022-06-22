from enum import auto
from glasses.models.vision.classification import AutoModelForClassification
from glasses.models.vision.classification.dummy.config import (
    DummyForClassificationConfig,
)
from glasses.storage import LocalStorage
from dataclasses import asdict

name = "dummy-d0"

# model, config = auto_model.from_name(name)


# auto_model.storage.put(name, model.state_dict(), asdict(config))
config: DummyForClassificationConfig = AutoModelForClassification.names_to_configs[name]
model, config = AutoModelForClassification.from_name(name, config)
print(model)
