# from glasses.models.vision.auto import AutoModelForClassification

from glasses.models.vision.backbones.dummy import DummyConfig
from glasses.models.vision.classification.common.model import AnyModelForClassification
from glasses.models.vision.classification.dummy import DummyForClassificationConfig
from glasses.models.vision.classification.heads.linear_head import LinearHeadConfig


model = DummyForClassificationConfig(
    backbone_config=DummyConfig(), head_config=LinearHeadConfig(64, 10)
).build()


# import importlib


# # module = importlib.import_module('.', "glasses.models.vision.classification.dummy.model")

# # AutoModelForClassification
# # print(AutoModelForClassification)
