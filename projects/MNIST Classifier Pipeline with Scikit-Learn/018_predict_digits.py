import numpy as np

def predict_digits(model, images):
    X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    return [int(p) for p in model.predict(X)]