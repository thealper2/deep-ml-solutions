import jax
import jax.numpy as jnp

def training_step(params, x, one_hot_targets, learning_rate):
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, learning_rate)
    return new_params, loss
