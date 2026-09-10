import numpy as np

def predict_images(pipeline, images):
    X = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    return [int(v) for v in pipeline.predict(X)]