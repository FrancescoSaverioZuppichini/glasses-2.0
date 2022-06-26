from typing import Any, Dict, Optional, List
from glasses.config import Config
import difflib
from torch import nn
from glasses.logger import logger
from glasses.storage import Storage, LocalStorage


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

    names_to_configs: Dict[str, Config]
    """Holds the map from name to config type"""

    @classmethod
    def get_config_from_name(cls, name: str) -> Config:
        return cls.names_to_configs[name]

    @classmethod
    def from_name(cls, name: str):
        if name not in cls.names_to_configs:
            suggestions = difflib.get_close_matches(name, cls.names_to_configs.keys())
            msg = f'Model "{name}" does not exists.'
            if len(suggestions) > 0:
                msg += f' Did you mean "{suggestions[0]}?"'
            raise KeyError(msg)

        config = cls.names_to_configs[name]
        return config.build()

    @classmethod
    def from_pretrained(
        cls, name: str, config: Optional[Config] = None, storage: Storage = None
    ) -> nn.Module:
        storage = LocalStorage() if storage is None else storage
        model = cls.from_name(name) if config is None else config.build()
        state_dict, _ = storage.get(name)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(str(e))
        logger.info(f"Loaded pretrained weights for {name}.")
        return model, config
