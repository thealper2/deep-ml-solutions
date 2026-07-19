def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    if num_transitions == 0:
        return {
            'observations': torch.zeros((0, 1, room_size, room_size), dtype=torch.float32),
            'actions': torch.zeros((0,), dtype=torch.long),
            'next_observations': torch.zeros((0, 1, room_size, room_size), dtype=torch.float32),
            'states': torch.zeros((0, 2), dtype=torch.float32),
            'next_states': torch.zeros((0, 2), dtype=torch.float32)
        }
        
    rng = torch.Generator()
    rng.manual_seed(seed)

    observations = []
    actions = []
    next_observations = []
    states = []
    next_states = []

    state, obs = env_reset(room_size, seed)

    for _ in range(num_transitions):
        action = torch.randint(0, 4, (1,), generator=rng).item()
        next_state, next_obs = env_step(state, action, room_size)

        observations.append(obs)
        actions.append(action)
        next_observations.append(next_obs)
        states.append(state)
        next_states.append(next_state)

        state = next_state
        obs = next_obs

    return {
        'observations': torch.stack(observations),
        'actions': torch.tensor(actions, dtype=torch.long),
        'next_observations': torch.stack(next_observations),
        'states': torch.stack(states),
        'next_states': torch.stack(next_states),
    }
