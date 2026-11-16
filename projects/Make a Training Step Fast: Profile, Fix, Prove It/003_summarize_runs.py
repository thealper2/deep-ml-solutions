def summarize_runs(times, warmup=2):
    """Drop the warmup runs, then return n, median, q1, q3, iqr and mean."""
    kept = times[warmup:]
    n = len(kept)
    if n == 0:
        return {
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "mean": None,
        }
    
    arr = np.asarray(kept, dtype=float)
    median = float(np.percentile(arr, 50))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    mean = float(arr.mean())

    return {
        "n": int(n),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mean": mean,
    }