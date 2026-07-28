def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    beta = 1 / (1 + np.exp(-(x @ params['wb'] + params['bb'])))
    z = x @ params['Wd1'] @ params['Wd2'] + params['ba']
    return beta, z
