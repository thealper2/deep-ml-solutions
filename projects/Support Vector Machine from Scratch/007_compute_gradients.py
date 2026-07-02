import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    scores = compute_scores(x, params)
    n_samples = len(y)

    dw = np.zeros_like(params['w'])
    db = 0.0

    for xi, yi, score in zip(x, y, scores):
        if yi * score < 1:
            dw += -yi * xi
            db += -yi

    dw = dw / n_samples + 2 * reg_lambda * params['w']
    db = db / n_samples
    return {'dw': dw, 'db': db}
