import numpy as np

class Tensor:

    def __init__(self, data: np.ndarray, requires_grad: bool = False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data: np.ndarray = data

        self.requires_grad: bool = requires_grad

        self.grad = None
        self._grad_fn = None
    
    @property
    def shape(self):
        return self.data.shape
    
    def __str__(self) -> str:
        return f"Tensor({str(self.data)})"
    
    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise Exception("This tensor doesn't require grad.")
        return self._grad_fn
    
    def backward(self, grad = None):
        if not self.requires_grad:
            raise Exception("This tensor doesn't require grad.")

        if not self.grad_fn:
            return False
        
        if grad is None and self.grad is None:
            grad = Tensor(1., requires_grad=False)
        
        elif self.grad is not None:
            grad = self.grad
        
        self.grad_fn.backward(grad)
        return True
    
    def add_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
    
    def __add__(self, o):
        if self.data is not None:
            self.data += o.data
        else:
            self.data = o.data
        return self