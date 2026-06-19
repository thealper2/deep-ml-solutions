def adam_bias_correct(moment, beta, t):
    return moment / (1 - beta ** t)
