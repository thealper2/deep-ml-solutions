import numpy as np


GROUP_SIZE = 32


def quantize(W, bits):
    """Group-wise symmetric quantization with float16 scales."""

    if bits != 4:
        raise ValueError("This quantizer is designed for 4-bit quantization.")

    out_features, in_features = W.shape
    flat = W.astype(np.float32).ravel()

    n = flat.size
    n_groups = (n + GROUP_SIZE - 1) // GROUP_SIZE

    qmax = 2 ** (bits - 1) - 1  # 7
    qmin = -2 ** (bits - 1)      # -8

    codes = np.empty(n, dtype=np.int8)
    scales = np.empty(n_groups, dtype=np.float16)

    for g in range(n_groups):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, n)

        group = flat[start:end]

        max_abs = np.max(np.abs(group))

        if max_abs == 0:
            scale = np.float32(1.0)
        else:
            scale = max_abs / qmax

        scales[g] = np.float16(scale)

        # Reconstruct using the actual stored float16 scale.
        scale32 = np.float32(scales[g])

        if scale32 == 0:
            codes[start:end] = 0
        else:
            q = np.round(group / scale32)
            q = np.clip(q, qmin, qmax)
            codes[start:end] = q.astype(np.int8)

    return {
        "codes": codes.reshape(W.shape),
        "scales": scales,
    }


def dequantize(q):
    """Reconstruct the original matrix from the quantized representation."""

    codes = q["codes"]
    scales = q["scales"]

    flat_codes = codes.ravel()
    n = flat_codes.size

    W_hat = np.empty(n, dtype=np.float32)

    for g, scale in enumerate(scales):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, n)

        W_hat[start:end] = (
            flat_codes[start:end].astype(np.float32)
            * np.float32(scale)
        )

    return W_hat.reshape(codes.shape)
