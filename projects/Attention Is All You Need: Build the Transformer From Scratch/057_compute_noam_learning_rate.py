def compute_noam_learning_rate(step, d_model, warmup_steps):
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5)) 
