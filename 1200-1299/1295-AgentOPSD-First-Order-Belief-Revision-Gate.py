import numpy as np

def first_order_delta_B(B_prev, c_prev, e, gamma: float = 1.0) -> float:
    """B_prev*(1-B_prev) * (e - (1-gamma)*c_prev)."""
    return B_prev * (1.0 - B_prev) * (e - (1 - gamma) * c_prev)
