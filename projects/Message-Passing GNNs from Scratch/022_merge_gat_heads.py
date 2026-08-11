def merge_gat_heads(head_outputs, mode='concat'):
    if isinstance(head_outputs, (list, tuple)):
        stacked = torch.stack(head_outputs, dim=0)
    else:
        stacked = head_outputs

    if mode == "concat":
        H, N, F = stacked.shape
        merged = stacked.permute(1, 0, 2).reshape(N, H * F)
    elif mode == "mean":
        merged = stacked.mean(dim=0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return stacked
