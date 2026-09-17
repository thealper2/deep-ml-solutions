import torch
import torch.nn as nn


def set_forget_bias(lstm, value):
    hidden = lstm.hidden_size
    with torch.no_grad():
        lstm.bias_ih_l0[hidden:2 * hidden] = value / 2
        lstm.bias_hh_l0[hidden:2 * hidden] = value / 2
    return lstm


def mean_forget_gate(lstm, x):
    hidden = lstm.hidden_size
    W_if = lstm.weight_ih_l0[hidden:2 * hidden, :]
    W_hf = lstm.weight_hh_l0[hidden:2 * hidden, :]
    b_if = lstm.bias_ih_l0[hidden:2 * hidden]
    b_hf = lstm.bias_hh_l0[hidden:2 * hidden]

    with torch.no_grad():
        outputs, (h_n, c_n) = lstm(x)

    T = x.shape[1]
    gate_vals = []
    for t in range(T):
        x_t = x[0, t, :]
        if t == 0:
            h_prev = torch.zeros(hidden)
        else:
            h_prev = outputs[0, t - 1, :]
        pre = W_if @ x_t + b_if + W_hf @ h_prev + b_hf
        gate_vals.append(torch.sigmoid(pre))
    gate_vals = torch.stack(gate_vals)
    return round(float(gate_vals.mean().item()), 4)


def gradient_to_first_input(lstm, x):
    x_clone = x.clone().detach().requires_grad_(True)
    outputs, _ = lstm(x_clone)
    loss = outputs[0, -1, :].sum()
    loss.backward()
    grad = x_clone.grad[0, 0, :]
    return float(grad.norm().item())


def horizon_curve(lstm, x, biases):
    results = []
    for b in biases:
        set_forget_bias(lstm, b)
        g = gradient_to_first_input(lstm, x)
        results.append(round(g, 6))
    return results