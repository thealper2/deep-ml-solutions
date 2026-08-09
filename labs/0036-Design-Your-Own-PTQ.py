import numpy as np
# numpy only


def ptq(weights, calib_X):
    """
    Design your own post-training quantization (PTQ) scheme.

    You receive FP32 weights of an MLP trained on real MNIST (Google/TF
    mnist.npz), plus a calibration batch of real flattened digit pixels
    in [0, 1]. Return dequantized weights after your PTQ round-trip.

    Args:
        weights: dict[str, np.ndarray]
            Keys: "W1","b1","W2","b2","W3","b3"
            Shapes: W1 (128,784), b1 (128,), W2 (64,128), b2 (64,), W3 (10,64), b3 (10,)
            Layout: each Wi is (out_features, in_features)
        calib_X: np.ndarray, shape (n_calib, 784)
            Real MNIST train pixels flattened, scaled to [0, 1].

    Returns:
        dequant_weights: dict[str, np.ndarray]
            Same keys/shapes, float dtype — low-bit reconstructions, not FP32 copies.
    """
    def quantize_tensor(x, bits=8, symmetric=True):
        """
        Quantize and dequantize a tensor with per-tensor symmetric quantization.
        """
        if symmetric:
            max_abs = np.max(np.abs(x))
            if max_abs == 0:
                return x.copy()
            qmax = 2 ** (bits - 1) - 1
            scale = max_abs / qmax
            q = np.clip(np.round(x / scale), -qmax, qmax)
            x_hat = q * scale
            return x_hat.astype(np.float32)
        else:
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max - x_min < 1e-12:
                return x.copy()
            qmin = 0
            qmax = 2 ** bits - 1
            scale = (x_max - x_min) / (qmax - qmin)
            zero_point = np.round(qmin - x_min / scale)
            q = np.clip(np.round(x / scale + zero_point), qmin, qmax)
            x_hat = scale * (q - zero_point)
            return x_hat.astype(np.float32)
    
    def quantize_weights(w, bits=8):
        """
        Quantize a weight matrix with per-channel quantization for better accuracy.
        """
        out_features, in_features = w.shape
        x_hat = np.zeros_like(w)
        
        for i in range(out_features):
            row = w[i, :]
            max_abs = np.max(np.abs(row))
            if max_abs == 0:
                x_hat[i, :] = row
                continue
            qmax = 2 ** (bits - 1) - 1
            scale = max_abs / qmax
            q = np.clip(np.round(row / scale), -qmax, qmax)
            x_hat[i, :] = q * scale
        
        return x_hat
    
    def quantize_bias(b, bits=8):
        """
        Bias tensors: per-tensor asymmetric quantization with higher precision.
        """
        return quantize_tensor(b, bits=bits, symmetric=False)
    
    dequant_weights = {}
    
    for name in ['W1', 'W2', 'W3']:
        dequant_weights[name] = quantize_weights(weights[name], bits=8)
    
    for name in ['b1', 'b2', 'b3']:
        dequant_weights[name] = quantize_bias(weights[name], bits=8)
    
    return dequant_weights
