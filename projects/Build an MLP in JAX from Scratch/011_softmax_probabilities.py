import jax.numpy as jnp

def softmax_probabilities(logits):
    unnormalized = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)
