from typing import Any, Callable

import torch


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd: Callable):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return self.lambd(x, *args, **kwargs)
