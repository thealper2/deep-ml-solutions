import jax.numpy as jnp

def log_softmax_logits(logits):
    c = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - c
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    return shifted_logits - log_sum_exp
