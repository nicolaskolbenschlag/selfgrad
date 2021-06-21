import abc
import numpy as np
import selfgrad

class Operator(abc.ABC):

    @abc.abstractmethod
    def forward(self) -> selfgrad.Tensor:
        pass

    @abc.abstractmethod
    def backward(self, grad: selfgrad.Tensor) -> None:
        pass

    def __call__(self, *args) -> selfgrad.Tensor:
        self.out = self.forward(args)
        self.out._grad_fn = self
        return self.out

class add(Operator):
    
    def forward(self, x: selfgrad.Tensor, y: selfgrad.Tensor) -> selfgrad.Tensor:
        self.x, self.y = x, y
        return selfgrad.Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad)
    
    def backward(self, grad: selfgrad.Tensor) -> None:
        if self.x.requires_grad:
            if self.x.shape == grad.shape:
                self.x.add_grad(grad)
            else:
                axis = np.argmax(np.abs(np.array(self.x.shape) - np.array(grad.shape)))
                self.x.add_grad(selfgrad.Tensor(grad.data.sum(axis=axis, keepdims=True)))
            
            self.x.backward()
        
        if self.y.requires_grad:
            if self.y.shape == grad.shape:
                self.y.add_grad(grad)
            else:
                axis = np.argmax(np.abs(np.array(self.y.shape) - np.array(grad.shape)))
                self.y.add_grad(selfgrad.Tensor(grad.data.sum(axis=axis, keepdims=True)))
            
            self.y.backward()

class mul(Operator):

    def forward(self, x: selfgrad.Tensor, y: selfgrad.Tensor) -> selfgrad.Tensor:
        self.x, self.y = x, y
        return selfgrad.Tensor(x.data * y.data, requires_grad=self.x.requires_grad or self.y.requires_grad)
    
    def backward(self, grad: selfgrad.Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(selfgrad.Tensor(grad.data * self.y.data))
            self.x.backward()
        
        if self.y.requires_grad:
            self.y.add_grad(selfgrad.Tensor(grad.data * self.x.data))
            self.y.backward()

class exp(Operator):

    def forward(self, x: selfgrad.Tensor) -> selfgrad.Tensor:
        self.x = x
        return selfgrad.Tensor(np.exp(x.data), requires_grad=self.x.requires_grad)
    
    def backward(self, grad: selfgrad.Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(selfgrad.Tensor(grad.data * np.exp(self.x.data)))
            self.x.backward()

class matmul(Operator):

    def forward(self, x: selfgrad.Tensor, y: selfgrad.Tensor) -> selfgrad.Tensor:
        self.x, self.y = x, y
        return selfgrad.Tensor(x.data @ y.data, requires_grad=self.x.requires_grad or self.y.requires_grad)
    
    def backward(self, grad: selfgrad.Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(selfgrad.Tensor(grad.data @ self.y.data.T))
            self.x.backward()
        
        if self.y.requires_grad:
            self.y.add_grad(selfgrad.Tensor(grad.data @ self.x.data.T))
            self.y.backward()

class sigmoid(Operator):

    def forward(self, x: selfgrad.Tensor) -> selfgrad.Tensor:
        self.x = x
        return selfgrad.Tensor(self.__class__.sigmoid(x.data), requires_grad=self.x.requires_grad)
    
    def backward(self, grad: selfgrad.Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(selfgrad.Tensor(grad.data * self.__class__.sigmoid(self.x.data) * (1 - self.__class__.sigmoid(self.x.data))))
            self.x.backward()
    
    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

class tanh(Operator):

    def forward(self, x: selfgrad.Tensor) -> selfgrad.Tensor:
        self.x = x
        return selfgrad.Tensor(np.tanh(x.data), requires_grad=self.x.requires_grad)
    
    def backward(self, grad: selfgrad.Tensor) -> None:
        if self.x.requires_grad:
            self.x.add_grad(selfgrad.Tensor(grad.data * (1 - np.tanh(self.x.data) ** 2)))
            self.x.backward()