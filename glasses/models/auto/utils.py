import importlib
from pathlib import Path
from typing import Dict, Iterator, Optional, List
from glasses.config import ConfigMixin, Config
import inspect
from glasses.logger import logger
from glasses.models import ModelZoo

CONFIG_VAR_NAME = "CONFIG"


def iter_models_modules(
    package: str, ignore_dirs: Optional[List[str]] = None
) -> Iterator[str]:
    ignore_dirs = [] if ignore_dirs is None else ignore_dirs
    # the following folders will be skipeed by default
    ignore_dirs += ["auto", "heads", "__pycache__", "common"]
    configs_to_models = {}
    # we import the package
    module = importlib.import_module(".", package)
    if not module:
        raise ModuleNotFoundError(f"{package} doesn't exist.")
    # and we get the path to the folder it's contained
    module_path = Path(module.__file__).parent
    # then we iterate all the subdirs and we look for packages to import
    for file_or_dir in module_path.iterdir():
        # if we have found a dir and it's not in ignore
        is_valid_dir = file_or_dir.is_dir() and file_or_dir.stem not in ignore_dirs
        if is_valid_dir:
            has_a_model = (file_or_dir / "model.py").exists()
            if has_a_model:
                yield f"{package}.{file_or_dir.stem}"


def get_configs_to_models_map(*args, **kwargs):
    configs_to_models = {}
    for module in iter_models_modules(*args, **kwargs):
        submodule = importlib.import_module(".", f"{module}.model")
        config, model = None, None
        non_module_specific_var = filter(
            lambda x: not x[0].startswith("__"), vars(submodule).items()
        )
        for name, value in non_module_specific_var:
            # check if this variables is the one that is suppose to hold the config map
            if name == CONFIG_VAR_NAME:
                config = value
            # check if we have a subclass of `ConfigMixin`, if yes, that will be mapped to our config
            if inspect.isclass(value):
                if issubclass(value, ConfigMixin):
                    model = value
        configs_to_models[config] = model

    return configs_to_models


def get_names_to_configs_map(*args, **kwargs):
    names_to_models_map = {}
    for module in iter_models_modules(*args, **kwargs):
        submodule = importlib.import_module(".", f"{module}.zoo")
        try:
            zoo = vars(submodule)["zoo"]
        except KeyError:
            raise KeyError(
                f"A `zoo.py` was found in {module} but no `zoo` was defined."
            )
        names_to_models_map = {**names_to_models_map, **zoo}

    return names_to_models_map
