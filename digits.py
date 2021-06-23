import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

from selfgrad import Tensor, Module, Parameter, SGD, Adam, Sigmoid

digits = sklearn.datasets.load_digits()

_, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training %i" % label)
plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
labels = sklearn.preprocessing.label_binarize(digits.target, list(range(10)))
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=.5, shuffle=False)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
y_test = y_test.argmax(axis=1)

class NeuralNetwork(Module):

    def __init__(self, n_input: int, n_hidden: int, n_out: int) -> None:
        self.fc_1 = Parameter(n_input, n_hidden)
        self.fc_2 = Parameter(n_hidden, n_out)
        self.b_2 = Parameter(1, n_out)

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.fc_1
        x = Sigmoid()(x)
        x = x @ self.fc_2 + self.b_2
        x = Sigmoid()(x)
        return x

model = NeuralNetwork(64, 32, 10)

# optim = SGD(model, .001)
optim = Adam(model)
batch_size = 32
epochs = 100

def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return ((y_true - y_pred) ** 2).mean()

losses, accuracies = [], []
for epoch in range(1, epochs + 1):
    
    loss = 0.
    
    for i in range(0, X_train.shape[0], batch_size):
        X, y = X_train[i : i + batch_size], y_train[i : i + batch_size]
        X, y = Tensor(X), Tensor(y)

        # NOTE would throw error because shape of gradients would not match
        if len(X) != batch_size:
            continue

        y_pred = model(X)
        batch_loss = mse(y, y_pred)

        loss += batch_loss.item

        model.zero_grad()
        batch_loss.backward()
        optim.step()
    
    y_test_pred = model(Tensor(X_test)).item.argmax(axis=1)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
    print(f"Epoch {epoch}: loss {round(loss, 2)} accuracy {round(accuracy, 2)}")
    losses += [loss]; accuracies += [accuracy]

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(losses, color="blue")
ax1.set_xlabel("Epoch")
ax1.set_title("Loss")
ax2.plot(accuracies, color="orange")
ax2.set_xlabel("Epoch")
ax2.set_title("Accuracy")
plt.show()

_, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax in axes:
    idx = np.random.randint(0, len(X_test))
    image, label = X_test[idx].reshape((8, 8)), y_test[idx]
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Predicted: %i" % label)
plt.show()