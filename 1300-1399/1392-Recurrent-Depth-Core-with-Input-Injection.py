import torch
import torch.nn as nn


def recurrent_depth_forward(e, s0, adapter, core, n_loops, return_all=False):
    e = torch.as_tensor(e, dtype=torch.float32)
    s = torch.as_tensor(s0, dtype=torch.float32)

    if n_loops == 0:
        return [] if return_all else s

    states = []
    for _ in range(n_loops):
        inp = torch.cat([e, s], dim=-1)
        s = core(adapter(inp))
        states.append(s)

    if return_all:
        return states

    return s