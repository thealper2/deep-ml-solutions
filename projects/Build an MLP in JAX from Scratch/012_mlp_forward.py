import jax
import jax.numpy as jnp

def mlp_forward(params, x):
    activation = x

    for layer in params[:-1]:
        outputs = linear_forward(activation, layer)
        activation = relu_activation(outputs)

    final_layer = params[-1]
    logits = linear_forward(activation, final_layer)
    return logits
