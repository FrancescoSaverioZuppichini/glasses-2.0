from .config import DummyConfig
from glasses.models import ModelZoo

zoo = ModelZoo(dummy_d0=DummyConfig(), dummy_d1=DummyConfig(out_channels=128))
