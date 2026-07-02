def inventory_skewed_quotes(fair_value, spread_width, inventory, skew_strength):
    half_spread = spread_width / 2.0
    shift = -inventory * skew_strength

    return {
        'bid': fair_value - half_spread + shift,
        'ask': fair_value + half_spread + shift,
    }
