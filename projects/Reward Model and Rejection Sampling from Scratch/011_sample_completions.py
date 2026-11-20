import torch
import torch.nn.functional as F


def sample_completions(model, prompt, n, temperature=1.0, generator=None):
    """n completions of len(prompt) tokens each, sampled token by token; temperature 0 is greedy."""
    model.eval()
    ids = encode_example(prompt)[0]
    seq = torch.tensor([ids] * n, dtype=torch.long)

    with torch.no_grad():
        for _ in range(len(prompt)):
            logits = model(seq)[:, -1, :]
            if temperature == 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)

    start = len(prompt) + 2
    return ["".join(CHARS[i] for i in row[start:].tolist()) for row in seq]