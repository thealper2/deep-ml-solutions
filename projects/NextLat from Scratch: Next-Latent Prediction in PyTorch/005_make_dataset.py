import torch
import numpy as np

def make_dataset(n: int, G: int, T: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    
    tokens_list = []
    masks_list = []
    states_list = []
    
    for _ in range(n):
        start_row = rng.integers(0, G)
        start_col = rng.integers(0, G)
        goal_row = rng.integers(0, G)
        goal_col = rng.integers(0, G)
        
        start = (start_row, start_col)
        goal = (goal_row, goal_col)
        
        max_len = T - 3
        moves = random_walk_to_goal(start, goal, G, max_len, rng)
        
        tokens, mask = encode_sequence(start, goal, moves, G, T)
        
        pos = start
        states = torch.zeros(T, dtype=torch.long)
        
        states[0] = start[0] * G + start[1]
        
        states[1] = start[0] * G + start[1]
        
        for i, move in enumerate(moves):
            pos, _ = grid_step(pos, move, G)
            states[i + 2] = pos[0] * G + pos[1]
        
        last_pos_idx = pos[0] * G + pos[1]
        for t in range(len(moves) + 2, T):
            states[t] = last_pos_idx
        
        tokens_list.append(tokens)
        masks_list.append(mask)
        states_list.append(states)
    
    return {
        'tokens': torch.stack(tokens_list),
        'mask': torch.stack(masks_list),
        'states': torch.stack(states_list),
        'G': G
    }