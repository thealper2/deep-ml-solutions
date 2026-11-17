def lr_width_fit(widths, log2_lrs, new_width):
    """Slope and intercept of log2(best lr) against log2(width), and the prediction at new_width."""
    x = np.log2(np.array(widths, dtype=float))
    y = np.array(log2_lrs, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * math.log2(new_width)

    return {
        "slope": round(float(slope), 3),
        "intercept": round(float(intercept), 3),
        "predicted_log2_lr": round(float(predicted), 3),
    }