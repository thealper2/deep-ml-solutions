def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    np.random.seed(seed)
    bandits = np.random.normal(loc=mean, scale=std, size=k)
    return bandits
