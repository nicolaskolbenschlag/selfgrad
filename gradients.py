from selfgrad import Tensor

w = Tensor(3., True)
a = Tensor(4.)
b = Tensor(1.)

y_pred = w * a + b
y = Tensor(7.)

loss = (y - y_pred) ** 2

print(f"y_pred: {y_pred}")
print(f"loss: {loss}")

loss.backward()

print(f"w.grad: {w.grad}")