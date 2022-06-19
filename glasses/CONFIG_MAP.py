from glasses.models.vision.classification.heads.LinearHead import (
    LinearHeadConfig,
    LinearHead,
)

from glasses.models.vision.classification.heads.stupid import (
    StupidHeadConfig,
    StupidHead,
)


from glasses.models.vision.backbones.dummy.config import DummyConfig
from glasses.models.vision.backbones.dummy.model import Dummy


CONFIG_MAP = {LinearHeadConfig: LinearHead, DummyConfig: Dummy, StupidHeadConfig: StupidHead}
