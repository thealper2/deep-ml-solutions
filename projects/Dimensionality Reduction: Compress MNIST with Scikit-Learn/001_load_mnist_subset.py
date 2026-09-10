import os
import tempfile
import urllib.request
import numpy as np

def load_mnist_subset(n=3000):
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            path,
        )

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(n, -1).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y