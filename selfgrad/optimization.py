import abc
import numpy as np

from selfgrad.module import Module

class Optimizer(abc.ABC):

    def __init__(self, module: Module) -> None:
        self.module = module

    @abc.abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        self.module.zero_grad()

class SGD(Optimizer):

    def __init__(self, module: Module, lr: float) -> None:
        super().__init__(module)
        self.lr = lr
    
    def step(self) -> None:
        for parameter in self.module.parameters:
            parameter.data -= parameter.grad.data * self.lr

class Adam(Optimizer):

    def __init__(self, module: Module, alpha: float = .001, beta1: float = .9, beta2: float = .999, eps: float = 1e-8) -> None:
        super().__init__(module)
        self.alpha, self.beta1, self.beta2, self.eps = alpha, beta1, beta2, eps
        self.m, self.v = [0. for _ in range(self.module.count_parameters())], [0. for _ in range(self.module.count_parameters())]
        self.t = 0
    
    def step(self) -> None:
        self.t += 1

        for idx, parameter in enumerate(self.module.parameters):
            
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * parameter.grad.data
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (parameter.grad.data ** 2)

            m = self.m[idx] / (1 - self.beta1 ** self.t)
            v = self.v[idx] / (1 - self.beta2 ** self.t)

            parameter.data -= self.alpha * m / (np.sqrt(v) + self.eps)