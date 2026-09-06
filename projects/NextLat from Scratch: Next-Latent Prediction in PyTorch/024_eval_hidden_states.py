import torch

def eval_hidden_states(dataset: dict, params: dict, n_heads: int, n_rows: int):
    tokens = dataset['tokens'][:n_rows]
    mask = dataset['mask'][:n_rows]

    x = tokens[:, :-1]
    mask_x = mask[:, :-1]

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)

    N = int(mask_x.sum().item())
    h_flat = h.reshape(-1, h.shape[-1])
    mask_flat = mask_x.reshape(-1)
    return h_flat[mask_flat]