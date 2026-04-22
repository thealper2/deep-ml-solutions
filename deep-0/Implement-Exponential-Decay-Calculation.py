def compute_exponential_decay_learning_rate(initial_learning_rate, decay_rate, current_step):
    lr = initial_learning_rate * (decay_rate ** current_step)
    return lr