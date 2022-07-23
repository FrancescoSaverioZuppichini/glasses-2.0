from abc import ABC, abstractmethod
from typing import List, Tuple

from glasses.config import Config
from glasses.types import StateDict


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, state_dict: StateDict, config: Config):
        pass

    @abstractmethod
    def get(self, key: str) -> Tuple[StateDict, Config]:
        pass

    @property
    @abstractmethod
    def models(self) -> List[str]:
        pass

    def __contains__(self, key: str) -> bool:
        return key in self.models
