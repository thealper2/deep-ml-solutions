import math

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""
    def stable_softmax(x):
        max_val = max(x)
        exp_x = [math.exp(val - max_val) for val in x]
        total = sum(exp_x)
        return [val / total for val in exp_x]
    
    stable = stable_softmax(scores)
    
    try:
        naive = []
        exp_x = [math.exp(val) for val in scores]
        total = sum(exp_x)
        naive = [val / total for val in exp_x]
        max_abs_diff = max(abs(n - s) for n, s in zip(naive, stable))
    except OverflowError:
        naive = [float('nan')] * len(scores)
        max_abs_diff = float('nan')
    
    return {
        'naive': [round(v, 6) if not math.isnan(v) else v for v in naive],
        'stable': [round(v, 6) for v in stable],
        'max_abs_diff': max_abs_diff if math.isnan(max_abs_diff) else round(max_abs_diff, 6)
    }