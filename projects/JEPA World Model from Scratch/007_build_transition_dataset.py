def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    return collect_random_transitions(num_transitions, room_size, seed)
