import torch

def merge_heads_and_project_output(context, w_o, b_o):
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)
