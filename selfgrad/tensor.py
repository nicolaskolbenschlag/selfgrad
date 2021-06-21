from __future__ import annotations
import numpy as np
import abc

class Tensor:

    def __init__(self, data: np.ndarray, requires_grad: bool = False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data: np.ndarray = data

        self.requires_grad: bool = requires_grad

        self.grad: Tensor = None
        self._grad_fn: Operator = None
    
    @property
    def shape(self):
        return self.data.shape
    
    def __str__(self) -> str:
        return f"Tensor{'*' if self.requires_grad else ''}({str(self.data)})"
    
    @property
    def grad_fn(self) -> Operator:
        if not self.requires_grad:
            raise Exception("This tensor doesn't require grad.")
        return self._grad_fn
    
    def backward(self, grad = None) -> bool:
        if not self.requires_grad:
            raise Exception("This tensor doesn't require grad.")

        if not self.grad_fn:
            return False
        
        if grad is None and self.grad is None:
            grad = Tensor(1.)
        
        elif self.grad is not None:
            grad = self.grad
        
        self.grad_fn.backward(grad)
        return True
    
    def add_grad(self, grad: Tensor) -> None:
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
    
    def zero_grad(self) -> None:
        self.grad = None
    
    def __add__(self, o: Tensor) -> Tensor:
        return Add()(self, o)
    
    def __mul__(self, o: Tensor) -> Tensor:
        return Mul()(self, o)

    def __sub__(self, o: Tensor) -> Tensor:
        return (Tensor(-1.) * self) + o
    
    def __pow__(self, e: int) -> Tensor:
        out = Tensor(1.)
        for _ in range(e):
            out *= self
        return out

    def __matmul__(self, o: Tensor) -> Tensor:
        return MatMul()(self, o)
    
    def sum(self, axis: int = None) -> Tensor:
        return Sum()(self, axis)
    
    def mean(self, axis: int = None) -> Tensor:
        raise NotImplementedError
    
    @property
    def item(self) -> np.ndarray:
        return self.data
    
    def detach(self) -> None:
        self.requires_grad = False

class Operator(abc.ABC):

    @abc.abstractmethod
    def forward(self) -> Tensor:
        pass

    @abc.abstractmethod
    def backward(self, grad: Tensor) -> None:
        pass

    def __call__(self, *args) -> Tensor:
        self.out = self.forward(*args)
        self.out._grad_fn = self
        return self.out

class Add(Operator):
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.x, self.y = x, y
        return Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad)
    
    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            if self.x.shape == grad.shape:
                self.x.add_grad(grad)
            else:
                axis = np.argmax(np.abs(np.array(self.x.shape) - np.array(grad.shape)))
                self.x.add_grad(Tensor(grad.data.sum(axis=axis, keepdims=True)))
            
            self.x.backward()
        
        if self.y.requires_grad:
            if self.y.shape == grad.shape:
                self.y.add_grad(grad)
            else:
                axis = np.argmax(np.abs(np.array(self.y.shape) - np.array(grad.shape)))
                self.y.add_grad(Tensor(grad.data.sum(axis=axis, keepdims=True)))
            
            self.y.backward()

class Sum(Operator):

    def forward(self, x: Tensor, axis: int = None) -> Tensor:
        self.x = x
        return Tensor(np.sum(x.data, axis=axis), requires_grad=x.requires_grad)

    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(Tensor(grad.data * np.ones_like(self.x)))
            self.x.backward()

class Mul(Operator):

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.x, self.y = x, y
        return Tensor(x.data * y.data, requires_grad=self.x.requires_grad or self.y.requires_grad)
    
    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(Tensor(grad.data * self.y.data))
            self.x.backward()
        
        if self.y.requires_grad:
            self.y.add_grad(Tensor(grad.data * self.x.data))
            self.y.backward()

class Exp(Operator):

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return Tensor(np.exp(x.data), requires_grad=self.x.requires_grad)
    
    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(Tensor(grad.data * np.exp(self.x.data)))
            self.x.backward()

class MatMul(Operator):

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.x, self.y = x, y
        return Tensor(x.data @ y.data, requires_grad=self.x.requires_grad or self.y.requires_grad)
    
    # TODO fix gradients
    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(Tensor(grad.data @ self.y.data.T))
            self.x.backward()
        
        if self.y.requires_grad:
            self.y.add_grad(Tensor(grad.data @ self.x.data.T))
            self.y.backward()

class Sigmoid(Operator):

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return Tensor(self.__class__.sigmoid(x.data), requires_grad=self.x.requires_grad)
    
    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(Tensor(grad.data * self.__class__.sigmoid(self.x.data) * (1 - self.__class__.sigmoid(self.x.data))))
            self.x.backward()
    
    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

class Tanh(Operator):

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return Tensor(np.tanh(x.data), requires_grad=self.x.requires_grad)
    
    def backward(self, grad: Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(Tensor(grad.data * (1 - np.tanh(self.x.data) ** 2)))
            self.x.backward()