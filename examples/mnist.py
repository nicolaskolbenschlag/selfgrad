import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

digits = sklearn.datasets.load_digits()

_, axes = plt.subplots(1, 4, figsize=(10, 3))

for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training %i" % label)

plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, digits.target, test_size=0.5, shuffle=False)