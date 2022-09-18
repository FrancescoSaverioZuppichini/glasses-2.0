from dataclasses import dataclass
from functools import partial

import pytest
from torch import Tensor, nn

from glasses.config import Config
from glasses.models.auto import AutoModel


class TestModel(nn.Module):
    """
    A very boring model, used for testing.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


@dataclass
class TestConfig(Config):
    """
    Equally boring config, used for testing :)
    """
    in_channels: int = 1
    out_channels: int = 2

    def build(self):
        return TestModel(**self.__dict__)


class TestAutoModel(AutoModel):
    """
    We also need to have an `AutoModel` for testing.
    """
    names_to_configs = {"test1": TestConfig, "test2": partial(TestConfig, 2)}


@pytest.fixture
def test_config():
    return TestConfig()


@pytest.fixture
def test_model_func():
    return TestModel


@pytest.fixture
def test_auto_model():
    return TestAutoModel
