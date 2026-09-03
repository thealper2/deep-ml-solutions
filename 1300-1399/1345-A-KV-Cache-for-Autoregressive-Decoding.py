import torch
import torch.nn.functional as F

class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)

        return self.k, self.v

    def __len__(self):
        if self.k is None:
            return 0
        return self.k.shape[2]

    def reset(self):
        self.k = None
        self.v = None

def attend_with_cache(q, k_new, v_new, cache):
    """
    Append new keys/values to cache, then attend q over the full cache.
    """
    k, v = cache.append(k_new, v_new)
    d = q.shape[-1]
    scale = d ** 0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output