from dataclasses import dataclass
import pytest
from torch import nn
from glasses.config import Config
from glasses.models.auto import AutoModel


class TestModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.fc(x)


@dataclass
class TestConfig(Config):
    in_channels: int = 1
    out_channels: int = 2

    def build(self):
        return TestModel(**self.__dict__)


class TestAutoModel(AutoModel):
    names_to_configs = {"test1": TestConfig(), "test2": TestConfig(2)}


@pytest.fixture
def test_config():
    return TestConfig()


@pytest.fixture
def test_model_func():
    return TestModel


@pytest.fixture
def test_auto_model():
    return TestAutoModel
