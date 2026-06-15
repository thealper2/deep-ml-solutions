def make_master_params(params):
    return {k: v.copy().astype(np.float32) for k, v in params.items()}
