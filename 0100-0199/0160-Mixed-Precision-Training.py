import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        self.loss_scale = loss_scale

    def forward(self, weights, inputs, targets):
        weights_f16 = weights.astype(np.float16)
        inputs_f16 = inputs.astype(np.float16)
        targets_f16 = targets.astype(np.float16)

        predictions = np.dot(inputs_f16, weights_f16)
        errors = predictions - targets_f16
        mse = np.mean(errors ** 2)

        scaled_loss = mse * self.loss_scale
        return float(scaled_loss)

    def backward(self, gradients):
        grads_f32 = np.array(gradients, dtype=np.float32)
        unscaled_grads = grads_f32 / self.loss_scale

        if np.any(np.isnan(unscaled_grads)) or np.any(np.isinf(unscaled_grads)):
            return np.zeros_like(unscaled_grads, dtype=np.float32)

        return unscaled_grads