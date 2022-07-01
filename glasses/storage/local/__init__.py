import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

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

    def put(self, key: str, state_dict: StateDict, config: Config):
        save_dir = self.root / Path(key)
        save_dir.mkdir(exist_ok=True)
        model_save_path = save_dir / f"model.{self.fmt}"
        config_save_path = save_dir / f"config.pkl"

        if key not in self or self.override:
            torch.save(state_dict, model_save_path)
            with open(config_save_path, "wb") as f:
                pickle.dump(config, f)
            assert model_save_path.exists()
            assert config_save_path.exists()

    def get(self, key: str) -> Tuple[StateDict, Config]:
        save_dir = self.root / Path(key)
        model_save_path = save_dir / f"model.{self.fmt}"
        config_save_path = save_dir / f"config.pkl"
        state_dict = torch.load(model_save_path)
        with open(config_save_path, "rb") as f:
            config = pickle.load(f)
        return state_dict, config

    @property
    def models(self) -> List[str]:
        return [file.stem for file in self.root.glob(f"*.{self.fmt}")]
