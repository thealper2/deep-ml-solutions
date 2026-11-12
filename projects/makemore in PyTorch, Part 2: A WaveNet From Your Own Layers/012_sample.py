import torch
import torch.nn.functional as F

def sample(model, itos, n, seed, block_size=8):
    """Sample n names in evaluation mode, each from a context of block_size '.' tokens."""
    for l in model.layers:
        if hasattr(l, 'training'):
            l.training = False
    g = torch.Generator()
    g.manual_seed(seed)
    names = []
    with torch.no_grad():
        for _ in range(n):
            ctx = [0] * block_size
            chars = []
            while True:
                X = torch.tensor([ctx], dtype=torch.long)
                logits = model(X)[0]
                probs = F.softmax(logits, dim=0)
                nxt = int(torch.multinomial(probs, 1, generator=g).item())
                if nxt == 0:
                    break
                chars.append(itos[nxt])
                ctx = ctx[1:] + [nxt]
            names.append("".join(chars))
    for l in model.layers:
        if hasattr(l, 'training'):
            l.training = True
    return names