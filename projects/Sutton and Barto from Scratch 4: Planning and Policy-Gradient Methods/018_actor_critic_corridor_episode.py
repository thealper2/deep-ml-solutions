import numpy as np


def actor_critic_corridor_episode(theta, w, alpha_theta, alpha_w, gamma, rng, max_steps=10000):
    th = np.array(theta, dtype=float, copy=True)
    wv = np.array(w, dtype=float, copy=True)

    state = 0
    discount = 1.0
    total_reward = 0.0

    for _ in range(max_steps):
        probs = corridor_policy(th)
        action = int(rng.choice(2, p=probs))
        next_state, reward, done = corridor_step(state, action)

        v_next = 0.0 if done else wv[next_state]
        delta = reward + gamma * v_next - wv[state]

        wv[state] += alpha_w * delta

        grad = -probs
        grad[action] += 1.0
        th = th + alpha_theta * discount * delta * grad
        discount *= gamma

        total_reward += reward

        if done:
            break
        state = next_state

    return th, wv, float(total_reward)