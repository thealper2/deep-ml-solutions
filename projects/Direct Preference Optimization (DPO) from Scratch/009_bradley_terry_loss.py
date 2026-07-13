def bradley_terry_loss(reward_chosen, reward_rejected):
    margin = reward_chosen - reward_rejected
    loss = -np.log(1 / (1 + np.exp(-margin)))
    return -np.mean(np.log(1 / (1 + np.exp(-margin))))
