import numpy as np


def reinforce_baseline_episode(theta, w, alpha_theta, alpha_w, gamma, rng, max_steps=10000):
    th = np.array(theta, dtype=float, copy=True)
    wv = float(w)

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
        delta = returns[t] - wv
        wv += alpha_w * delta
        probs = corridor_policy(th)
        grad = -probs
        grad[actions[t]] += 1.0
        th = th + alpha_theta * (gamma ** t) * delta * grad

    total_reward = float(np.sum(rewards))
    return th, wv, total_reward