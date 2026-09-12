import os
import tempfile
import urllib.request
import numpy as np

def load_mnist(n_train=10000, n_test=2000):
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            path
        )

    with np.load(path) as d:
        X_train = d["x_train"][:n_train].reshape(n_train, -1).astype(np.float32)
        y_train = d["y_train"][:n_train].astype(np.int64)
        X_test = d["x_test"][:n_test].reshape(n_test, -1).astype(np.float32)
        y_test = d["y_test"][:n_test].astype(np.int64)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }