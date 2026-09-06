import torch

def valid_move_rate(dataset: dict, params: dict, n_heads: int, n_rows: int) -> float:
    tokens = dataset['tokens'][:n_rows]
    mask = dataset['mask'][:n_rows]
    states = dataset['states'][:n_rows]
    G = dataset['G']
    EOS = 4 + G * G

    x = tokens[:, :-1]
    y_mask = mask[:, 1:]
    states_before = states[:, :-1]

    goal_idx = tokens[:, 1] - 4
    goal_idx_expanded = goal_idx.unsqueeze(1).expand(-1, states_before.shape[1])

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        logits = output_head(h, params)
        preds = torch.argmax(logits, dim=-1)

    valid = 0
    total = 0
    T_minus_1 = x.shape[1]
    for i in range(x.shape[0]):
        for t in range(1, T_minus_1):
            if not y_mask[i, t]:
                continue
            total += 1
            pos_cell = int(states_before[i, t].item())
            goal_cell = int(goal_idx_expanded[i, t].item())
            pred = int(preds[i, t].item())

            if pos_cell == goal_cell:
                if pred == EOS:
                    valid += 1
            else:
                row = pos_cell // G
                col = pos_cell % G
                if pred in legal_actions((row, col), G):
                    valid += 1

    if total == 0:
        return 0.0
    return valid / total