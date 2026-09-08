import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch

def load_fashion_mnist(n_train=10000, n_test=2000):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    temp_dir = tempfile.gettempdir()
    data = {}

    for key, filename in files.items():
        filepath = os.path.join(temp_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)

        with gzip.open(filepath, "rb") as f:
            raw_data = f.read()
        
        if "images" in key:
            parsed = np.frombuffer(raw_data, dtype=np.uint8, offset=16)
            num_images = parsed.shape[0] // (28 * 28)
            parsed = parsed.reshape(num_images, 28, 28)
        else:
            parsed = np.frombuffer(raw_data, dtype=np.uint8, offset=8)

        data[key] = parsed

    return {
        "X_train": torch.tensor(data["train_images"][:n_train], dtype=torch.float32) / 255.0,
        "y_train": torch.tensor(data["train_labels"][:n_train], dtype=torch.int64),
        "X_test": torch.tensor(data["test_images"][:n_test], dtype=torch.float32) / 255.0,
        "y_test": torch.tensor(data["test_labels"][:n_test], dtype=torch.int64),
    }