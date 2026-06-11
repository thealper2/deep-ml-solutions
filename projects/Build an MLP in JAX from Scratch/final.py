"""
Build an MLP in JAX from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  make_prng_key ──
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    return jax.random.PRNGKey(seed)

# ── Step 002  split_prng_key ──
import jax

def split_prng_key(key, num):
    return jax.random.split(key, num=num)

# ── Step 003  sample_normal_matrix ──
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    return jax.random.normal(key, shape=shape)

# ── Step 004  sample_input_features ──
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    return jax.random.normal(key, shape=(batch_size, num_features))

# ── Step 005  assign_class_labels ──
import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    mask = jnp.arange(inputs.shape[1]) >= num_classes
    masked_arr = jnp.where(mask, -jnp.inf, inputs)
    result = jnp.argmax(masked_arr, axis=1)
    return result

# ── Step 006  one_hot_encode_labels ──
import jax

def one_hot_encode_labels(labels, num_classes):
    one_hot_matrix = jax.nn.one_hot(labels, num_classes)
    return one_hot_matrix

# ── Step 007  init_linear_layer ──
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    weights = jax.random.normal(key, shape=(in_dim, out_dim)) * scale
    biases = jnp.zeros((out_dim,))
    return {'W': weights, 'b': biases}

# ── Step 008  init_mlp_params ──
import jax
import jax.numpy as jnp

def init_mlp_params(key, layer_sizes, scale=0.1):
    params = []
    keys = jax.random.split(key, num=len(layer_sizes) - 1)

    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        layer_params = init_linear_layer(keys[i], in_dim, out_dim, scale)
        params.append(layer_params)

    return params

# ── Step 009  linear_forward ──
import jax.numpy as jnp

def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    return x @ layer_params['W'] + layer_params['b']

# ── Step 010  relu_activation ──
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    return jnp.maximum(0, x)

# ── Step 011  softmax_probabilities ──
import jax.numpy as jnp

def softmax_probabilities(logits):
    unnormalized = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)

# ── Step 012  mlp_forward ──
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

# ── Step 013  log_softmax_logits ──
import jax.numpy as jnp

def log_softmax_logits(logits):
    c = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - c
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    return shifted_logits - log_sum_exp

# ── Step 014  cross_entropy_loss ──
import jax
import jax.numpy as jnp

def cross_entropy_loss(logits, one_hot_targets):
    log_probs = log_softmax_logits(logits)
    target_indices = jnp.argmax(one_hot_targets, axis=1)
    correct_log_probs = jnp.take_along_axis(log_probs, target_indices[:, None], axis=1)
    return -jnp.mean(correct_log_probs)

# ── Step 015  classification_accuracy ──
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    predictions = jnp.argmax(logits, axis=1)
    correct = (predictions == labels)
    return jnp.mean(correct)

# ── Step 016  loss_fn_of_params ──
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)

# ── Step 017  compute_param_grads ──
import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    grad_fn = jax.grad(loss_fn_of_params, argnums=0)
    return grad_fn(params, x, one_hot_targets)

# ── Step 018  sgd_update_params ──
import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    new_params = []
    for layer_params, layer_grads in zip(params, grads):
        new_W = layer_params['W'] - learning_rate * layer_grads['W']
        new_b = layer_params['b'] - learning_rate * layer_grads['b']
        new_params.append({'W': new_W, 'b': new_b})
    
    return new_params

# ── Step 019  training_step ──
import jax
import jax.numpy as jnp

def training_step(params, x, one_hot_targets, learning_rate):
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, learning_rate)
    return new_params, loss

# ── Step 020  train_mlp ──
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    current_params = params
    for _ in range(num_epochs):
        current_params, _ = training_step(current_params, x, one_hot_targets, learning_rate)

    return current_params

# ── Step 021  predict_classes ──
def predict_classes(params, x):
    logits = mlp_forward(params, x)
    return jnp.argmax(logits, axis=1)

# ── Scaffold (runner) ──
"""End-to-end demo: train a small MLP in JAX from scratch on synthetic data."""
import numpy as np
import jax
import jax.numpy as jnp

from solution import (
    make_prng_key,
    split_prng_key,
    sample_normal_matrix,
    sample_input_features,
    assign_class_labels,
    one_hot_encode_labels,
    init_linear_layer,
    init_mlp_params,
    linear_forward,
    relu_activation,
    softmax_probabilities,
    mlp_forward,
    log_softmax_logits,
    cross_entropy_loss,
    classification_accuracy,
    loss_fn_of_params,
    compute_param_grads,
    sgd_update_params,
    training_step,
    train_mlp,
    predict_classes,
)


if __name__ == "__main__":
    np.random.seed(0)

    # --- Config ---
    seed = 0
    batch_size = 32
    num_features = 8
    num_classes = 4
    hidden_size = 16
    learning_rate = 0.1
    num_epochs = 200

    # --- PRNG setup ---
    root_key = make_prng_key(seed)
    data_key, init_key = split_prng_key(root_key, 2)

    # --- Synthetic data ---
    x = sample_input_features(data_key, batch_size, num_features)
    labels = assign_class_labels(x, num_classes)
    y_onehot = one_hot_encode_labels(labels, num_classes)

    print("Input shape:", x.shape)
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())
    print("One-hot[0]:", np.asarray(y_onehot[0]).tolist())

    # --- Init MLP ---
    layer_sizes = [num_features, hidden_size, hidden_size, num_classes]
    params = init_mlp_params(init_key, layer_sizes, scale=0.1)
    print("Num layers:", len(params))
    print(
        "Layer shapes:",
        [(np.asarray(W).shape, np.asarray(b).shape) for (W, b) in params],
    )

    # --- Forward pass before training ---
    logits0 = mlp_forward(params, x)
    loss0 = cross_entropy_loss(logits0, y_onehot)
    acc0 = classification_accuracy(logits0, labels)
    print(f"Initial loss:     {float(loss0):.4f}")
    print(f"Initial accuracy: {float(acc0):.4f}")

    # --- Train ---
    trained_params = train_mlp(params, x, y_onehot, learning_rate, num_epochs)

    # --- Evaluate ---
    logits_final = mlp_forward(trained_params, x)
    loss_final = cross_entropy_loss(logits_final, y_onehot)
    acc_final = classification_accuracy(logits_final, labels)
    preds = predict_classes(trained_params, x)

    print(f"Final loss:       {float(loss_final):.4f}")
    print(f"Final accuracy:   {float(acc_final):.4f}")
    print("Preds[:8]: ", np.asarray(preds[:8]).tolist())
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())
