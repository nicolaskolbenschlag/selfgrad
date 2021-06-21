import abc
import selfgrad

class Optimizer(abc.ABC):

    def __init__(self, module: selfgrad.Module) -> None:
        self.module = module

    @abc.abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        self.module.zero_grad()

class SGD(Optimizer):

    def __init__(self, module: selfgrad.Module, lr: float) -> None:
        super().__init__(module)
        self.lr = lr
    
    def step(self) -> None:
        for parameter in self.module.parameters:
            parameter.data -= parameter.grad.data * self.lr