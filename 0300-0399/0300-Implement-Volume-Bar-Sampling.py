import numpy as np

def volume_bars_sampling(prices: np.ndarray, volumes: np.ndarray, volume_threshold: float) -> list:
    """
    Generate volume bars from tick/trade data.
    
    Args:
        prices: Array of trade prices for each trade/tick
        volumes: Array of trade volumes corresponding to each price
        volume_threshold: Volume threshold that triggers formation of a new bar
    
    Returns:
        List of bars, where each bar is [open, high, low, close, total_volume]
        All values rounded to 4 decimal places
    """
    if len(prices) == 0 or len(volumes) == 0 or volume_threshold <= 0:
        return []

    bars = []
    i = 0
    n = len(prices)

    while i < n:
        bar_open = float(prices[i])
        bar_high = float(prices[i])
        bar_low = float(prices[i])
        bar_close = float(prices[i])
        bar_volume = 0.0

        while i < n and bar_volume < volume_threshold:
            bar_high = max(bar_high, prices[i])
            bar_low = min(bar_low, prices[i])
            bar_close = float(prices[i])
            bar_volume += float(volumes[i])
            i += 1

        bars.append([
            round(bar_open, 4),
            round(bar_high, 4),
            round(bar_low, 4),
            round(bar_close, 4),
            round(bar_volume, 4),
        ])

    return bars