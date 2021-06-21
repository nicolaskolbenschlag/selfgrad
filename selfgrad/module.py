import typing
import numpy as np
import inspect
import selfgrad

class Parameter(selfgrad.Tensor):
    
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)

class Module:

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