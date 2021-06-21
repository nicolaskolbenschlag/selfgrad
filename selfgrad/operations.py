import abc
import selfgrad

class Operation(abc.ABC):

    @abc.abstractmethod
    def forward(self) -> selfgrad.Tensor:
        pass

    @abc.abstractmethod
    def backward(self):
        pass

    def __call__(self, *args) -> selfgrad.Tensor:
        self.out = self.forward(args)
        self.out._grad_fn = self
        return self.out

class Add(Operation):
    pass