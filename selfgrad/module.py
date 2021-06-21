import abc
import typing
import numpy as np
import inspect

from selfgrad.tensor import Tensor

class Parameter(Tensor):
    
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)

class Module(abc.ABC):

    @property
    def parameters(self) -> typing.Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    
    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()
    
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)