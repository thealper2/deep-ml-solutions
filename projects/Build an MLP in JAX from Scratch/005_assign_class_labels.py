import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    mask = jnp.arange(inputs.shape[1]) >= num_classes
    masked_arr = jnp.where(mask, -jnp.inf, inputs)
    result = jnp.argmax(masked_arr, axis=1)
    return result
