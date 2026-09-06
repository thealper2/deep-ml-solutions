import numpy as np
import torch

def detour_robustness(params: dict, n_heads: int, G: int, max_steps: int, n_trials: int,
                      detour_prob: float = 0.75, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    successes = 0
    EOS = 4 + G * G
    
    for _ in range(n_trials):
        start_row = rng.integers(0, G)
        start_col = rng.integers(0, G)
        goal_row = rng.integers(0, G)
        goal_col = rng.integers(0, G)
        
        start = (start_row, start_col)
        goal = (goal_row, goal_col)
        
        start_cell = 4 + start_row * G + start_col
        goal_cell = 4 + goal_row * G + goal_col
        seq = [start_cell, goal_cell]
        pos = start
        reached_goal = False
        
        for _ in range(max_steps):
            if pos == goal:
                reached_goal = True
                break
            
            if rng.random() < detour_prob:
                legal = legal_actions(pos, G)
                action = int(rng.choice(legal))
            else:
                preds = greedy_decode(params, n_heads, seq, 1)
                action = preds[0]
                
                if action < 0 or action > 3:
                    reached_goal = False
                    break
                
                legal = legal_actions(pos, G)
                if action not in legal:
                    reached_goal = False
                    break
            
            pos, legal = grid_step(pos, action, G)
            if not legal:
                reached_goal = False
                break
            
            seq.append(action)
        
        if pos == goal:
            successes += 1
    
    return successes / n_trials