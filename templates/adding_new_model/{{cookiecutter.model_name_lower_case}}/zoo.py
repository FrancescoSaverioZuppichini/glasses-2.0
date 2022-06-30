from glasses.models.auto import ModelZoo

from .config import {{cookiecutter.ConfigNameCammelCase}}


def {{cookiecutter.model_checkpoint}}():
    return {{cookiecutter.ConfigNameCammelCase}}()

zoo = ModelZoo({{cookiecutter.model_checkpoint}}={{cookiecutter.model_checkpoint}})