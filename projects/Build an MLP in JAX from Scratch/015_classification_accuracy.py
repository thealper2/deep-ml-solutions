import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    predictions = jnp.argmax(logits, axis=1)
    correct = (predictions == labels)
    return jnp.mean(correct)
