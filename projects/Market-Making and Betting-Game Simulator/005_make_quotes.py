def make_quotes(fair_value, spread_width):
    half_spread = spread_width / 2.0
    return {
        'bid': fair_value - half_spread,
        'ask': fair_value + half_spread,
    }
