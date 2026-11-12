import torch

def sample_bigram(P, itos, n, seed):
    """Sample n names, one generator seeded with `seed` for every draw."""
    g = torch.Generator()
    g.manual_seed(seed)
    names = []
    for _ in range(n):
        cur = 0
        chars = []
        while True:
            nxt = int(torch.multinomial(P[cur], 1, replacement=True, generator=g).item())
            if nxt == 0:
                break
            chars.append(itos[nxt])
            cur = nxt
        names.append("".join(chars))
    return names