from .LinearHead import LinearHeadConfig, LinearHead
from .stupid import StupidHeadConfig, StupidHead

CONFIGS_TO_MODELS = {
    LinearHeadConfig: LinearHead,
    StupidHeadConfig: StupidHead,
}
