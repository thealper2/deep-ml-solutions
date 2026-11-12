import torch
import torch.nn.functional as F

def sample_mlp(model, itos, n, seed):
    """Sample n names from the MLP, starting each from an all-'.' context."""
    g = torch.Generator()
    g.manual_seed(seed)
    bs = model.block_size
    names = []
    with torch.no_grad():
        for _ in range(n):
            ctx = [0] * bs
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
    return names