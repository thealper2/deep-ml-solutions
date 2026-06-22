import torch

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2
    
