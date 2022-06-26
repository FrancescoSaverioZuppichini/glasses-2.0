from glasses.models import ModelZoo

from .config import DummyConfig

zoo = ModelZoo(dummy_d0=DummyConfig(), dummy_d1=DummyConfig(out_channels=128))
