def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    arr = np.array([large_value])
    exp_arr = np.exp(arr)
    naive_exp = float(exp_arr[0])
    overflowed = np.isinf(naive_exp)
    return {'naive_exp': naive_exp, 'overflowed': overflowed}
