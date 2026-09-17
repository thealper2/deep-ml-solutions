import numpy as np


def _sigmoid(net):
    return 1.0 / (1.0 + np.exp(-net))


def activation_derivative(net, act):
    if act == 'sigmoid':
        s = _sigmoid(net)
        return s * (1 - s)
    elif act == 'tanh':
        t = np.tanh(net)
        return 1 - t ** 2
    else:
        raise ValueError(f"unknown activation: {act}")


def error_flow(w, nets, act):
    prod = 1.0
    for net in nets:
        prod *= float(activation_derivative(net, act)) * w
    return float(prod)


def critical_weight(act):
    if act == 'sigmoid':
        return 4.0
    elif act == 'tanh':
        return 1.0
    else:
        raise ValueError(f"unknown activation: {act}")


def steps_to_decay(factor, threshold):
    if factor >= 1.0:
        return None
    q = 1
    while factor ** q >= threshold:
        q += 1
    return q


def flow_curve(w, net, act, lags):
    return [round(error_flow(w, [net] * q, act), 6) for q in lags]