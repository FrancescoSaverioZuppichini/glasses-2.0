from dataclasses import dataclass
from glasses.config import Config
from .model import {{cookiecutter.ModelNameCammelCase}}

@dataclass
class {{cookiecutter.ConfigNameCammelCase}}(Config):
    param1: int = 10

    def built(self) -> {{cookiecutter.ModelNameCammelCase}}:
        return {{cookiecutter.ModelNameCammelCase}}(**self.__dict__)