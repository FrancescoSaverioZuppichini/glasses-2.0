import difflib
from typing import Callable, Dict, Optional

from torch import nn

from glasses.config import Config
from glasses.logger import logger
from glasses.storage import LocalStorage, Storage


class AutoModel:
    """The base `AutoModel` class.

    Usage:

    ```python

    auto_model = AutoModel()
    model = auto_model.from_name("my_name")
    model = auto_model.from_pretrained("my_name")
    model = auto_model.from_pretrained("my_name", my_config)
    ```
    """

    names_to_configs: Dict[str, Callable[[], Config]]
    """Holds the map from name to config type"""

    @classmethod
    def get_config_from_name(cls, name: str) -> Config:
        return cls.names_to_configs[name]()

    @classmethod
    def from_name(cls, name: str):
        if name not in cls.names_to_configs:
            suggestions = difflib.get_close_matches(name, cls.names_to_configs.keys())
            msg = f'Model "{name}" does not exists.'
            if len(suggestions) > 0:
                msg += f' Did you mean "{suggestions[0]}?"'
            raise KeyError(msg)

        config = cls.names_to_configs[name]()
        return config.build()

    @classmethod
    def from_pretrained(
        cls, name: str, config: Optional[Config] = None, storage: Storage = None
    ) -> nn.Module:
        storage = LocalStorage() if storage is None else storage
        state_dict, loaded_config = storage.get(name)
        config = loaded_config if config is None else config
        model = config.build()
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(str(e))
        logger.info(f"Loaded pretrained weights for {name}.")
        return model, config
