from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class ApplyToKeys:
    transform: Callable
    keys: List[str]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, val in data.items():
            if key in self.keys:
                data[key] = self.transform(val)
        return data
