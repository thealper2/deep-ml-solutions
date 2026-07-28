def situ_glu(x, Wg, Wu, beta1=4.0, beta2=25.0):
    """(softcap(x@Wg, b1) * sigmoid(x@Wg)) * softcap(x@Wu, b2), softcap = b*tanh(u/b).

    Use sigmoid(g) = 0.5*(1 + tanh(0.5*g)) to stay overflow-safe. |out| <= b1*b2.
    """
    g = x @ Wg
    u = x @ Wu

    softcap_g = beta1 * np.tanh(g / beta1)
    softcap_u = beta2 * np.tanh(u / beta2)

    sigmoid_g = 0.5 * (1 + np.tanh(0.5 * g))
    
    return softcap_g * sigmoid_g * softcap_u
