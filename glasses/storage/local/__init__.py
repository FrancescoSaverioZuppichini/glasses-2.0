import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

from glasses.config import Config
from glasses.types import StateDict

from ..base import Storage


@dataclass
class LocalStorage(Storage):
    root: Path = Path("/tmp/glasses")
    override: bool = False
    fmt: str = "pth"

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)

    def put(self, key: str, state_dict: StateDict, config: Dict):
        save_dir = self.root / Path(key)
        save_dir.mkdir(exist_ok=True)
        model_save_path = save_dir / f"model.{self.fmt}"
        config_save_path = save_dir / f"config.json"

        if key not in self or self.override:
            torch.save(state_dict, model_save_path)
            with open(config_save_path, "w") as f:
                json.dump(config, f)
            assert model_save_path.exists()
            assert config_save_path.exists()

    def get(self, key: str) -> Tuple[StateDict, Config]:
        save_dir = self.root / Path(key)
        model_save_path = save_dir / f"model.{self.fmt}"
        config_save_path = save_dir / f"config.json"
        state_dict = torch.load(model_save_path)
        with open(config_save_path, "r") as f:
            config = json.load(f)
        return state_dict, config

    @property
    def models(self) -> List[str]:
        return [file.stem for file in self.root.glob(f"*.{self.fmt}")]
