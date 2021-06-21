import numpy as np
import selfgrad

class Tensor:

    def __init__(self, data: np.ndarray, requires_grad: bool = False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data: np.ndarray = data

        self.requires_grad: bool = requires_grad

        self.grad: selfgrad.Tensor = None
        self._grad_fn: selfgrad.Operation = None
    
    @property
    def shape(self):
        return self.data.shape
    
    def __str__(self) -> str:
        return f"Tensor({str(self.data)})"
    
    @property
    def grad_fn(self) -> selfgrad.Operator:
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
    
    def add_grad(self, grad: selfgrad.Tensor) -> None:
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
    
    def zero_grad(self) -> None:
        self.grad = None
    
    def __add__(self, o: selfgrad.Tensor) -> selfgrad.Tensor:
        return selfgrad.add(self, o)
    
    def __mul__(self, o: selfgrad.Tensor) -> selfgrad.Tensor:
        return selfgrad.mul(self, o)
    
    def __matmul__(self, o: selfgrad.Tensor) -> selfgrad.Tensor:
        return selfgrad.matmul(self, o)