import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import matplotlib.pyplot as plt

from selfgrad import Tensor, Module, Parameter, SGD

digits = sklearn.datasets.load_digits()

_, axes = plt.subplots(1, 4, figsize=(10, 3))

for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training %i" % label)

# plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
labels = sklearn.preprocessing.label_binarize(digits.target, list(range(10)))
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.5, shuffle=False)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

class NeuralNetwork(Module):

    def __init__(self, n_input: int, n_hidden: int, n_out: int) -> None:
        self.fc_1 = Parameter(n_input, n_hidden)
        self.fc_2 = Parameter(n_hidden, n_out)
        self.b_2 = Parameter(n_out)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.fc_1
        x = x @ self.fc_2 + self.b_2
        return x

model = NeuralNetwork(64, 16, 10)

optim = SGD(model, .01)
batch_size = 32
epochs = 2

def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return ((y_true - y_pred) ** 2).sum()

for epoch in range(1, epochs + 1):
    
    loss = Tensor(0.)
    
    for i in range(0, X_train.shape[0], batch_size):
        X, y = X_train[i : i + batch_size], y_train[i : i + batch_size]
        X, y = Tensor(X), Tensor(y)

        model.zero_grad()
        y_pred = model(X)

        batch_loss = mse(y, y_pred)
        loss += batch_loss
    
    print(f"Epoch {epoch}: {loss.item}")

    loss.backward()
    # optim.step()