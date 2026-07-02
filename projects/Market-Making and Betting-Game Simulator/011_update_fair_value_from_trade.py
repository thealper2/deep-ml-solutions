def update_fair_value_from_trade(fair_value, side, bid, ask, adjustment):
    if adjustment == 0:
        return fair_value
    if side == 'buy':
        return fair_value + adjustment
    else:
        return fair_value - adjustment
