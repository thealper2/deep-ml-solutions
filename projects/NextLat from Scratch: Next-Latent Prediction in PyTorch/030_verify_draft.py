import torch

def verify_draft(params: dict, n_heads: int, prefix: list, next_token: int, drafts: list) -> tuple:
    seq = prefix + [next_token] + drafts
    tokens = torch.tensor([seq], dtype=torch.long)

    with torch.no_grad():
        h = gpt_hidden_states(tokens, params, n_heads)
        logits = output_head(h, params)
        preds = torch.argmax(logits, dim=-1).squeeze(0)

    base = len(prefix)
    n_accepted = 0
    for j, draft in enumerate(drafts):
        if int(preds[base + j].item()) == draft:
            n_accepted += 1
        else:
            break

    corr = int(preds[base + n_accepted].item())
    return n_accepted, corr