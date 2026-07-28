def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    rms = np.sqrt(np.mean(o ** 2, axis=1, keepdims=True) + 1e-6)
    o_norm = o / rms
    gate = 1 / (1 + np.exp(-(x @ Wg)))
    return (gate * o_norm) @ Wo
