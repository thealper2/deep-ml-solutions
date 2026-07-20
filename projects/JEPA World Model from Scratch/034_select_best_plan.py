def select_best_plan(action_sequences, costs):
    best_idx = torch.argmin(costs)
    return action_sequences[best_idx]
