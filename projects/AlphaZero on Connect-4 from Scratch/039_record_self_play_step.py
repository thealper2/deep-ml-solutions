def record_self_play_step(history, board, policy, to_play):
    history.append({
        'board': board.copy(),
        'policy': policy.copy(),
        'to_play': to_play,
    })
    return history
