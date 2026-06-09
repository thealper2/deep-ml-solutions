import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)
