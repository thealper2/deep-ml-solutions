def run_market_making_episode(true_value, counterparty_sides, initial_fair_value, config):
    base_spread = config.get('base_spread', 0.0)
    uncertainty = config.get('uncertainty', 0.0)
    skew_strength = config.get('skew_strength', 0.0)
    belief_adjustment = config.get('belief_adjustment', 0.0)
    
    state = {'cash': 0.0, 'inventory': 0.0}
    fair_value = initial_fair_value
    history = []
    
    for side in counterparty_sides:
        spread_width = uncertainty_spread(base_spread, uncertainty)
        quotes = inventory_skewed_quotes(fair_value, spread_width, state['inventory'], skew_strength)
        bid = quotes['bid']
        ask = quotes['ask']
        
        state = execute_trade(state, side, bid, ask)
        
        history.append({
            'bid': bid,
            'ask': ask,
            'side': side,
            'cash': state['cash'],
            'inventory': state['inventory'],
            'fair_value': fair_value
        })
        
        fair_value = update_fair_value_from_trade(fair_value, side, bid, ask, belief_adjustment)
    
    pnl = mark_to_market_pnl(state['cash'], state['inventory'], true_value)
    
    return {
        'pnl': pnl,
        'cash': state['cash'],
        'inventory': state['inventory'],
        'fair_value': fair_value,
        'history': history
    }
