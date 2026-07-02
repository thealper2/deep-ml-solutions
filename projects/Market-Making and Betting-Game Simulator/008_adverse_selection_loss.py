import numpy as np

def adverse_selection_loss(fair_value, bid, ask, informed_values, informed_probabilities):
    loss = 0.0
    for v, p in zip(informed_values, informed_probabilities):
        if v > ask:
            loss += p * (v - ask)
        elif v < bid:
            loss += p * (bid - v)

    return loss
