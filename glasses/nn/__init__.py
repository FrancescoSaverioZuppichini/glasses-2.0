from typing import Callable

from torch import Tensor, nn


class Lambda(nn.Module):
    def __init__(self, lambd: Callable[[Tensor], Tensor]):
        """An utility Module, it allows custom function to be passed
        Args:
            lambd (Callable[Tensor]): A function that does something on a tensor

        Usage:

        ```python
            add_two = Lambda(lambd x: x + 2)
            add_two(Tensor([0])) // 2
        ```

        """
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return self.lambd(x)
