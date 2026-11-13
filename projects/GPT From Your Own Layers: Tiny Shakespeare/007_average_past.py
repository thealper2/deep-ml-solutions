import torch
import torch.nn.functional as F

def average_past(x):
    """Position t of the output is the mean of x[:, :t + 1], computed with a masked softmax and a matrix multiply."""
    B, T, C = x.shape
    w = torch.zeros((T, T))
    w = w.masked_fill(torch.triu(torch.ones((T, T)), diagonal=1).bool(), float('-inf'))
    w = F.softmax(w, dim=-1)
    return w @ x