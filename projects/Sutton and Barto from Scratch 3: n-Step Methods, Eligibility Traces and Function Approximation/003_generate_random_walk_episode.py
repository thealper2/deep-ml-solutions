import numpy as np

def generate_random_walk_episode(n_states, rng, max_jump=1):
    state = n_states // 2
    states = []
    rewards = []
    while True:
        nxt, reward, done = random_walk_step(state, n_states, rng, max_jump)
        states.append(state)
        rewards.append(reward)
        if done:
            break

        state = nxt

    return (states, rewards)