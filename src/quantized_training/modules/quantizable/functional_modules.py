from typing import Union

import torch
from torch import Tensor

__all__ = ['AddFunctional', 'MulFunctional', 'Matmul']

class AddFunctional(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Union[Tensor, float]) -> Tensor:
        return torch.add(x, y)

class MulFunctional(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Union[Tensor, float]) -> Tensor:
        return torch.mul(x, y)
    
class MatmulFunctional(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.matmul(x, y)