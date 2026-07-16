import numpy as np

def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    loss, grads = dpo_loss_grad(params, batch, ref_logprobs_batch, beta)
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grads[key]

    metrics = {'loss': loss}
    return updated_params, metrics
