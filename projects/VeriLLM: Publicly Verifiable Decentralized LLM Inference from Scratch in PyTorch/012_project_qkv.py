import numpy as np

def project_qkv(x, attn_params):
    q = linear_projection(x, attn_params['Wq'], attn_params['bq'])
    k = linear_projection(x, attn_params['Wk'], attn_params['bk'])
    v = linear_projection(x, attn_params['Wv'], attn_params['bv'])
    return q, k, v
