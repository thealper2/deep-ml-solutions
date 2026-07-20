def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    N, H = action_sequences.shape
    D = start_embedding.shape[0]
    total_costs = torch.zeros(N)

    for seq_idx in range(N):
        current = start_embedding
        total_cost = 0.0

        for t in range(H):
            action = action_sequences[seq_idx, t]
            current = predict_next_embedding(
                current.unsqueeze(0),
                action.unsqueeze(0),
                predictor_params
            ).squeeze(0)
            cost = latent_cost(current, goal_embedding)
            total_cost += cost

        total_costs[seq_idx]= total_cost

    return total_costs
