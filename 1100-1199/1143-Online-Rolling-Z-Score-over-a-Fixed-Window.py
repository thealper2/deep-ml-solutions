def rolling_zscore(stream, window):
    if window <= 0:
        return []

    result = []
    buffer = []

    for i, value in enumerate(stream):
        buffer.append(value)

        if len(buffer) > window:
            buffer.pop(0)

        if len(buffer) < window:
            result.append(0.0)
            continue

        n = len(buffer)
        mean = sum(buffer) / n
        variance = sum((x - mean) ** 2 for x in buffer) / n

        if variance == 0:
            result.append(0.0)
        else:
            latest = buffer[-1]
            z_score = (latest - mean) / (variance ** 0.5)
            result.append(z_score)

    return result