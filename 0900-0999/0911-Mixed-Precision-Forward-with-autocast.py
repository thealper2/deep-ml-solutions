import torch

def forward_bf16(model, x):
    with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = model(x)

    return out.float()
