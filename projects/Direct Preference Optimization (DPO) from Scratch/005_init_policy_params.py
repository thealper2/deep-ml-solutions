def init_policy_params(vocab_size, d_model, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    embed = rng.standard_normal((vocab_size, d_model)) * 0.02
    W_out = rng.standard_normal((d_model, vocab_size)) * 0.02
    b_out = np.zeros(vocab_size)

    return {
        'embed': embed,
        'W_out': W_out,
        'b_out': b_out,
    }
