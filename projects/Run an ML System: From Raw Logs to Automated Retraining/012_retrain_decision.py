def retrain_decision(daily, floor=0.75, patience=2, cooldown=3, last_retrain=None):
    """First day the AUC has been below floor for `patience` entries in a row, or None."""
    streak = 0
    for day in sorted(daily.keys()):
        if last_retrain is not None and day < last_retrain + cooldown:
            streak = 0
            continue
        if daily[day] < floor:
            streak += 1
        else:
            streak = 0
        if streak >= patience:
            return day
    return None