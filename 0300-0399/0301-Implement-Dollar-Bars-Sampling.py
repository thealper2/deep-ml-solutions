def dollar_bars(trades: list, dollar_threshold: float) -> list:
    """
    Generate dollar bars from trade data.
    
    Args:
        trades: List of tuples (price, volume) representing individual trades
        dollar_threshold: Dollar amount threshold for creating a new bar
        
    Returns:
        List of dollar bars as tuples (open, high, low, close, volume, dollar_value)
    """
    if len(trades) == 0 or dollar_threshold <= 0:
        return []

    bars = []
    i = 0
    n = len(trades)

    while i < n:
        price, volume = trades[i]
        bar_open = price
        bar_high = price
        bar_low = price
        bar_close = price
        bar_volume = 0.0
        bar_dollar = 0.0

        while i < n and bar_dollar < dollar_threshold:
            price, volume = trades[i]
            trade_dollar = price * volume

            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_close = price
            bar_volume += volume
            bar_dollar += trade_dollar
            i += 1
        
        if bar_dollar >= dollar_threshold:
            bars.append((
                round(bar_open, 4),
                round(bar_high, 4),
                round(bar_low, 4),
                round(bar_close, 4),
                round(bar_volume, 2),
                round(bar_dollar, 2),
            ))

    return bars