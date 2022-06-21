from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from glasses.config import Config
from torch import nn
from glasses.types import StateDict


class Storage(ABC):
    @abstractmethod
    def put(self, state_dict: StateDict, config: Dict):
        pass

    @abstractmethod
    def get(self, key: str) -> Tuple[StateDict, Dict]:
        pass

    @property
    @abstractmethod
    def models(self) -> List[str]:
        pass

    def __contains__(self, key: str) -> bool:
        return key in self.models
