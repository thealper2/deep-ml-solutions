import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    n_states = transition_matrix.shape[0]

    sequence = [initial_state]
    current_state = initial_state

    for _ in range(num_steps):
        probs = transition_matrix[current_state]
        next_state = np.random.choice(n_states, p=probs)
        sequence.append(next_state)
        current_state = next_state

    return np.array(sequence)