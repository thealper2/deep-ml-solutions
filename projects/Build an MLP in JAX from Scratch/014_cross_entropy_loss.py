import jax
import jax.numpy as jnp

def cross_entropy_loss(logits, one_hot_targets):
    log_probs = log_softmax_logits(logits)
    target_indices = jnp.argmax(one_hot_targets, axis=1)
    correct_log_probs = jnp.take_along_axis(log_probs, target_indices[:, None], axis=1)
    return -jnp.mean(correct_log_probs)
