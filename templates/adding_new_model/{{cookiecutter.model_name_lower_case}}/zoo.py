from .config import {{cookiecutter.ConfigNameCammelCase}}

def {{cookiecutter.model_checkpoint}}():
    return {{cookiecutter.ConfigNameCammelCase}}()

zoo = dict({{cookiecutter.model_checkpoint}}={{cookiecutter.model_checkpoint}})