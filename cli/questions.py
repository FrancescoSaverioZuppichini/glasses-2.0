from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


class InputCollector:
    def __call__(self, value: str, default: Optional[str] = None):
        new_value = f"Select {value}"
        res = input(new_value)
        if res == "":
            if default is None:
                print("Empty input. Please try again.")
                return self.__call__(value)
            else:
                res = default
        return res


@dataclass
class Ask:
    what: str
    default: Optional[str] = None
    input_collector: InputCollector = field(default_factory=InputCollector)

    def __call__(self, context: Dict):
        res = self.input_collector(
            f"{self.what}:[{self.default}] ", default=self.default
        )
        context[self.what] = res


@dataclass
class Select:
    what: str
    paths: Dict
    input_collector: InputCollector = field(default_factory=InputCollector)

    def __call__(self, context: Dict):
        key = self.input_collector(f"{self.what}:[{','.join(self.paths.keys())}] ")
        if key not in self.paths:
            raise KeyError(f"{key} not in {list(self.paths.keys())}")

        context[self.what] = {key: {}}
        self.paths[key](context[self.what][key])


@dataclass
class Choices:
    what: str
    choices: List
    input_collector: InputCollector = field(default_factory=InputCollector)

    def __call__(self, context: Dict):
        idx_to_choices = {str(idx): val for idx, val in enumerate(self.choices)}
        print("\n".join([f"{idx}:{choice}" for idx, choice in idx_to_choices.items()]))
        idx = self.input_collector(f"{self.what}:[{','.join(idx_to_choices.keys())}] ")
        if idx not in idx_to_choices:
            raise IndexError
        context[self.what] = idx_to_choices[idx]


@dataclass
class IfHasSelected:
    selections: List[str]
    question: Callable

    def __call__(self, context: Dict):
        if any([selection in context.values() for selection in self.selections]):
            self.question(context)


@dataclass
class Questions:
    funcs: List[Callable]

    def __call__(self, context: Dict):
        for func in self.funcs:
            func(context)
