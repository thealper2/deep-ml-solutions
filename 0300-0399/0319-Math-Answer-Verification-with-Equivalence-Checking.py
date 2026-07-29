import re
import math

def verify_math_answer(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Verify if two mathematical answers are equivalent.
    
    Args:
        predicted: The predicted answer string
        ground_truth: The ground truth answer string
        tolerance: Numerical tolerance for comparison
    
    Returns:
        True if answers are equivalent, False otherwise
    """
    if predicted.strip() == ground_truth.strip():
        return True
    
    def safe_eval(expr):
        expr = expr.strip()
        if expr.startswith('sqrt(') and expr.endswith(')'):
            inner = expr[5:-1]
            return math.sqrt(safe_eval(inner))
        if expr == 'pi':
            return math.pi
        if '/' in expr and not expr.startswith('sqrt'):
            try:
                num, den = expr.split('/')
                return safe_eval(num) / safe_eval(den)
            except:
                pass
        try:
            return float(expr)
        except:
            try:
                return eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi, "e": math.e})
            except:
                return float('nan')
    
    pred_val = safe_eval(predicted)
    truth_val = safe_eval(ground_truth)
    
    if not math.isnan(pred_val) and not math.isnan(truth_val):
        return abs(pred_val - truth_val) <= tolerance
    
    return False
