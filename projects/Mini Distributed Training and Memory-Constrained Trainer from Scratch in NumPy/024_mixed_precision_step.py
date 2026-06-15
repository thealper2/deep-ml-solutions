def mixed_precision_step(x, y, master_params, scale, lr):
    master_copy = {}
    for k, v in master_params.items():
        master_copy[k] = v.copy().astype(np.float32)
    
    fp16_params = {}
    for k, v in master_copy.items():
        fp16_params[k] = v.astype(np.float16)
    
    x_fp16 = x.astype(np.float16)
    y_fp16 = y.astype(np.float16)
    y_pred, cache = mlp_forward(x_fp16, fp16_params)
    loss_fp16, dy_pred = mse_loss_and_grad(y_pred, y_fp16)
    dy_pred_scaled = dy_pred * scale
    fp16_grads = mlp_backward(dy_pred_scaled, cache, fp16_params)
    fp32_grads = {}
    for k, v in fp16_grads.items():
        fp32_grads[k] = v.astype(np.float32) / scale
    
    if has_non_finite_gradients(fp32_grads):
        return float(loss_fp16), master_params, True
    
    for k in master_copy:
        master_copy[k] -= lr * fp32_grads[k]

    return float(loss_fp16), master_copy, False
