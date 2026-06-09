def predict_classes(params, x):
    logits = mlp_forward(params, x)
    return jnp.argmax(logits, axis=1)
