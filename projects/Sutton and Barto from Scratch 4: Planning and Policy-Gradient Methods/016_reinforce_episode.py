import numpy as np


def reinforce_episode(theta, alpha, gamma, rng, max_steps=10000):
    th = np.array(theta, dtype=float, copy=True)

    state = 0
    actions = []
    rewards = []

    for _ in range(max_steps):
        probs = corridor_policy(th)
        action = int(rng.choice(2, p=probs))
        state, reward, done = corridor_step(state, action)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    T = len(rewards)
    returns = [0.0] * T
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    for t in range(T):
        probs = corridor_policy(th)
        grad = -probs
        grad[actions[t]] += 1.0
        th = th + alpha * (gamma ** t) * returns[t] * grad

    total_reward = float(np.sum(rewards))
    return th, total_reward