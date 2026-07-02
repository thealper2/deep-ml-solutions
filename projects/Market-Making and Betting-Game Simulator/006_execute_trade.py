def execute_trade(state, side, bid, ask, size=1):
    new_state = state.copy()
    if side == 'buy':
        new_state['cash'] += ask * size
        new_state['inventory'] -= size
    else:
        new_state['cash'] -= bid * size
        new_state['inventory'] += size
    
    return new_state
