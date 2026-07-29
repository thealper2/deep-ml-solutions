import torch
import torch.nn as nn


def dropout_demo():
    """Demonstrate Dropout behavior in eval vs train mode.

    Returns:
        tuple: (eval_output, train_nonzero_count)
            eval_output: result of Dropout(ones) in eval mode (identity)
            train_nonzero_count: int count of nonzero elements after Dropout in train mode
    """
    torch.manual_seed(0)
    x = torch.ones(10)
    drop = nn.Dropout(p=0.5)

    drop.eval()
    eval_output = drop(x)

    drop.train()
    train_output = drop(x)
    train_nonzero_count = int((train_output != 0).sum().item())

    return eval_output, train_nonzero_count
