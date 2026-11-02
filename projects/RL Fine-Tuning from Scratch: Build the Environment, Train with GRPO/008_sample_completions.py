import torch


@torch.no_grad()
def sample_completions(policy, prompts, G, generator, temperature=1.0):
    B = prompts.shape[0]
    seqs = prompts.repeat_interleave(G, 0)
    total = B * G

    finished = torch.zeros(total, dtype=torch.bool, device=seqs.device)
    mask = torch.zeros(total, COMP_LEN, dtype=torch.float, device=seqs.device)

    for step in range(COMP_LEN):
        logits = policy(seqs)[:, -1, :] / temperature
        probs = torch.softmax(logits, -1)
        draw = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        draw = torch.where(finished, torch.full_like(draw, PAD), draw)

        mask[:, step] = (~finished).float()

        finished = finished | (draw == EOS)

        seqs = torch.cat([seqs, draw.unsqueeze(1)], dim=1)

    return seqs, mask