import numpy as np

def symmetry_augmented_training(q_table, state_board, action, reward, next_state_board, done, alpha, gamma):
    """Apply Q-learning updates to all 8 D4 symmetries of a transition."""
    row, col = action // 3, action % 3

    def identity(r, c):
        return r, c

    def rot90(r, c):
        return 2 - c, r

    def compose(f, g):
        return lambda r, c: f(*g(r, c))

    rotations = [identity]
    for _ in range(3):
        rotations.append(compose(rot90, rotations[-1]))

    def fliplr(r, c):
        return r, 2 - c

    coord_transforms = list(rotations) + [compose(rot, fliplr) for rot in rotations]

    def apply_board_transforms(b):
        rots = [b]
        for _ in range(3):
            rots.append(np.rot90(rots[-1]))
        flipped = np.fliplr(b)
        frots = [flipped]
        for _ in range(3):
            frots.append(np.rot90(frots[-1]))
        return rots + frots

    state_variants = apply_board_transforms(state_board)
    next_variants = apply_board_transforms(next_state_board)

    for i, coord_fn in enumerate(coord_transforms):
        s_board = state_variants[i]
        ns_board = next_variants[i]
        new_row, new_col = coord_fn(row, col)
        new_action = new_row * 3 + new_col

        state_key = encode_board_state_key(s_board)

        if done:
            target = q_learning_terminal_target(reward)
        else:
            next_state_key = encode_board_state_key(ns_board)
            next_legal_actions = [r * 3 + c for r, c in get_legal_moves(ns_board)]
            target = q_learning_nonterminal_target(
                reward, gamma, q_table, next_state_key, next_legal_actions
            )

        old_q = get_q_value(q_table, state_key, new_action)
        new_q = old_q + alpha * (target - old_q)
        set_q_value(q_table, state_key, new_action, new_q)

    return q_table
