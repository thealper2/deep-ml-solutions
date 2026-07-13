def reward_accuracy(reward_chosen, reward_rejected):
    return np.mean(reward_chosen > reward_rejected)
