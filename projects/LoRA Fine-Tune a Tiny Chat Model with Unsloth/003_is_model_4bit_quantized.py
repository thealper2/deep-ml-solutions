import torch
import bitsandbytes as bnb

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            return True

    return False
