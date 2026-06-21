import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)
