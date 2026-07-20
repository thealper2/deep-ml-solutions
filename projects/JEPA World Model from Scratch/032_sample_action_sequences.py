def sample_action_sequences(n_sequences, horizon, n_actions):
    return torch.randint(0, n_actions, (n_sequences, horizon))
