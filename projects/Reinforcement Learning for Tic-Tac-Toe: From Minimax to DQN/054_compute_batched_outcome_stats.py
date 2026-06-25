import numpy as np

def compute_batched_outcome_stats(episode_outcomes, batch_size):
    """Aggregate outcomes into per-batch win/loss/draw rates."""
    win_rates = []
    loss_rates = []
    draw_rates = []
    batch_indices = []
    
    for i in range(0, len(episode_outcomes) - batch_size + 1, batch_size):
        batch = episode_outcomes[i:i+batch_size]
        n = len(batch)
        wins = batch.count('win')
        losses = batch.count('loss')
        draws = batch.count('draw')

        win_rates.append(wins / n)
        loss_rates.append(losses / n)
        draw_rates.append(draws / n)
        batch_indices.append(i // batch_size)

    return {
        'batch_index': np.array(batch_indices),
        'win_rate': np.array(win_rates),
        'loss_rate': np.array(loss_rates),
        'draw_rate': np.array(draw_rates),
    }
