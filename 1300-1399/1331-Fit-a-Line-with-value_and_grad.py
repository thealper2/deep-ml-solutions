import jax
import jax.numpy as jnp

def fit_line(x, y, lr, steps):
    """Gradient-descent fit of y ≈ w*x + b from w=b=0.
    Returns (w, b, final_loss) as Python floats."""
    w, b = 0.0, 0.0

    def loss(w, b):
        pred = w * x + b
        return jnp.mean((pred - y) ** 2)

    grad_loss = jax.value_and_grad(loss, argnums=(0, 1))

    for _ in range(steps):
        loss_val, (dw, db) = grad_loss(w, b)
        w = w - lr * dw
        b = b - lr * db

    final_loss, _ = grad_loss(w, b)

    return float(w), float(b), float(final_loss)