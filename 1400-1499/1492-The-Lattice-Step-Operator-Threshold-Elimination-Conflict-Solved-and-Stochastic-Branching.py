import torch
import torch.nn.functional as F


def eliminate(x, probs, theta):
    return x & (probs >= theta)


def has_conflict(x, cls_prob, theta_cls):
    empty_cell = ~x.any(dim=-1)                     # (B, n, n)
    any_empty = empty_cell.flatten(1).any(dim=-1)   # (B,)
    return any_empty | (cls_prob > theta_cls)


def is_solved(x):
    counts = x.sum(dim=-1)                          # (B, n, n)
    return (counts == 1).flatten(1).all(dim=-1)


def branch_pin(x, logits, tau, generator):
    B, n, _, V = x.shape
    new_x = x.clone()
    picks = torch.full((B, 3), -1, dtype=torch.long)

    for b in range(B):
        open_cells = [(r, c) for r in range(n) for c in range(n) if int(x[b, r, c].sum()) >= 2]
        if not open_cells:
            continue
        ci = int(torch.randint(0, len(open_cells), (1,), generator=generator).item())
        r, c = open_cells[ci]

        alive = x[b, r, c]
        cell_logits = logits[b, r, c] / tau
        masked = torch.where(alive, cell_logits, torch.full_like(cell_logits, float("-inf")))
        probs = F.softmax(masked, dim=-1)
        digit = int(torch.multinomial(probs, 1, generator=generator).item())

        new_x[b, r, c] = False
        new_x[b, r, c, digit] = True
        picks[b, 0] = r
        picks[b, 1] = c
        picks[b, 2] = digit

    return new_x, picks


def step_symbolic(x, logits, cls_logit, theta, theta_cls, tau, generator):
    probs = torch.sigmoid(logits)
    cls_prob = torch.sigmoid(cls_logit)

    eliminated = eliminate(x, probs, theta)
    conflict = has_conflict(eliminated, cls_prob, theta_cls)
    solved = is_solved(eliminated) & ~conflict

    new_x = eliminated.clone()
    picks = torch.full((x.shape[0], 3), -1, dtype=torch.long)

    branch_rows = [b for b in range(x.shape[0]) if not bool(solved[b]) and not bool(conflict[b])]
    if branch_rows:
        branched, sub = branch_pin(eliminated, logits, tau, generator)
        for b in branch_rows:
            new_x[b] = branched[b]
            picks[b] = sub[b]

    return new_x, conflict, solved, picks