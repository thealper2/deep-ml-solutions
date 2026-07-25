def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    preferences = preferences.copy()
    k = len(preferences)

    exp_pref = np.exp(preferences - np.max(preferences))
    probs = exp_pref / np.sum(exp_pref)

    advantage = reward - average_reward

    for a in range(k):
        if a == action:
            preferences[a] += alpha * advantage * (1 - probs[a])
        else:
            preferences[a] -= alpha * advantage * probs[a]

    return preferences
