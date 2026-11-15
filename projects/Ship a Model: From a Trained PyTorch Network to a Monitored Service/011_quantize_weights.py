import copy
import torch
from torch import nn


def quantize_weights(model, bits=8):
    """Copy of model with per-row symmetric fake-quantized Linear weights, and its size report."""
    q = copy.deepcopy(model)
    L = 2 ** (bits - 1) - 1

    float_bytes = 0
    quant_weights = 0
    quant_scales = 0
    biases = 0

    for module in q.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            rows = w.shape[0]
            row_max = w.abs().amax(dim=1, keepdim=True)
            scale = torch.where(row_max > 0, row_max / L, torch.ones_like(row_max))
            wq = torch.clamp(torch.round(w / scale), -L, L) * scale
            module.weight.data = wq

            n = w.numel()
            float_bytes += n
            quant_weights += n
            quant_scales += rows
            if module.bias is not None:
                b = module.bias.numel()
                float_bytes += b
                biases += b

    float_bytes *= 4
    quantized_bytes = (quant_weights * bits) // 8 + quant_scales * 4 + biases * 4
    ratio = round(float_bytes / quantized_bytes, 2)

    return q, {
        "float_bytes": float_bytes,
        "quantized_bytes": quantized_bytes,
        "ratio": ratio,
    }