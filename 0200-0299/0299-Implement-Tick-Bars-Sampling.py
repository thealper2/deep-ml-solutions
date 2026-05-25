def tick_bars(ticks: list, bar_size: int) -> list:
    """
    Sample tick data into tick bars.
    
    Args:
        ticks: List of tuples (timestamp, price, volume) representing individual trades
        bar_size: Number of ticks per bar
    
    Returns:
        List of dictionaries with keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    if not ticks or bar_size <= 0:
        return []

    bars = []
    num_bars = len(ticks) // bar_size

    for i in range(num_bars):
        start_idx = i * bar_size
        end_idx = start_idx + bar_size - 1

        bar_ticks = ticks[start_idx:end_idx+1]

        timestamps = [t[0] for t in bar_ticks]
        prices = [t[1] for t in bar_ticks]
        volumes = [t[2] for t in bar_ticks]

        bar = {
            'timestamp': timestamps[-1],
            'open': round(prices[0], 2),
            'high': round(max(prices), 2),
            'low': round(min(prices), 2),
            'close': round(prices[-1], 2),
            'volume': round(sum(volumes), 2),
        }
        bars.append(bar)

    return bars