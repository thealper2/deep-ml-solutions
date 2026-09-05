def output_head(h, params: dict):
    head_w = params['head_w']
    head_b = params['head_b']
    return h @ head_w + head_b