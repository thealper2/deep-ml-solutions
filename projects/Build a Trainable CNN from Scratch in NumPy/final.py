"""
Build a Trainable CNN from Scratch in NumPy — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  argmax_rows ──
def argmax_rows(matrix):
    return np.argmax(matrix, axis=1)

# ── Step 002  row_max ──
import numpy as np

def row_max(matrix):
    return np.max(matrix, axis=1, keepdims=True)

# ── Step 003  row_sum ──
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    return np.sum(matrix, axis=1, keepdims=True)

# ── Step 004  exp_shifted ──
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    max_values = row_max(logits)
    exp_values = np.exp(logits - max_values)
    return exp_values

# ── Step 005  stable_softmax ──
def stable_softmax(logits):
    shifted_exp = exp_shifted(logits)
    return shifted_exp / np.sum(shifted_exp, axis=1, keepdims=True)

# ── Step 006  one_hot ──
def one_hot(labels, num_classes):
    one_hot = np.eye(num_classes)[labels]
    return one_hot

# ── Step 007  gather_true_class_probs ──
def gather_true_class_probs(probs, labels):
    return np.array([prob[label] for prob, label in zip(probs, labels)])

# ── Step 008  cross_entropy_loss ──
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    probs = np.clip(probs, eps, 1.0 - eps)
    correct_probs = gather_true_class_probs(probs, labels)
    log_probs = np.log(correct_probs)
    loss = -np.mean(log_probs)
    if abs(loss) < eps:
        return -0.0

    return loss

# ── Step 009  accuracy ──
def accuracy(logits_or_probs, labels):
    y_pred = argmax_rows(logits_or_probs)
    correct = (y_pred == labels).sum()
    return correct / len(labels)

# ── Step 010  he_std ──
def he_std(fan_in):
    std_dev = np.sqrt(2.0 / fan_in)
    return std_dev

# ── Step 011  he_init ──
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std_dev = np.sqrt(2.0 / fan_in)
    weights_normal = np.random.randn(*shape) * std_dev
    return weights_normal

# ── Step 012  init_zero_bias ──
import numpy as np

def init_zero_bias(length):
    return np.zeros(length, dtype=np.float64)

# ── Step 013  pad_2d ──
def pad_2d(images, pad):
    if pad == 0:
        return images

    return np.pad(
        images,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant',
        constant_values=0
    )

# ── Step 014  output_spatial_size ──
def output_spatial_size(input_size, kernel, stride, padding):
    h_out = (input_size + 2 * padding - kernel) // stride + 1
    return h_out

# ── Step 015  im2col ──
def im2col(images, kernel_h, kernel_w, stride, padding):
    padded = pad_2d(images, padding)
    N, C, H_pad, W_pad = padded.shape

    H_out = output_spatial_size(images.shape[2], kernel_h, stride, padding)
    W_out = output_spatial_size(images.shape[3], kernel_w, stride, padding)

    patch_size = C * kernel_h * kernel_w
    num_patches = N * H_out * W_out
    patches = np.zeros((num_patches, patch_size), dtype=images.dtype)

    patch_idx = 0
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride
                h_end = h_start + kernel_h
                w_start = w_out * stride
                w_end = w_start + kernel_w

                patch = padded[n, :, h_start:h_end, w_start:w_end]
                patches[patch_idx] = patch.flatten()
                patch_idx += 1

    return patches

# ── Step 016  col2im ──
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape

    H_out = output_spatial_size(H, kernel_h, stride, padding)
    W_out = output_spatial_size(W, kernel_w, stride, padding)
    
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    output_pad = np.zeros((N, C, H_pad, W_pad))

    patch_size = C * kernel_h * kernel_w

    patch_idx = 0
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                patch = cols[patch_idx].reshape(C, kernel_h, kernel_w)

                h_start = h_out * stride
                h_end = h_start + kernel_h
                w_start = w_out * stride
                w_end = w_start + kernel_w

                output_pad[n, :, h_start:h_end, w_start:w_end] += patch
                patch_idx += 1

    if padding > 0:
        return output_pad[:, :, padding:-padding, padding:-padding]

    return output_pad

# ── Step 017  conv2d_forward ──
def conv2d_forward(x, weights, bias, stride, padding):
    N, C, H, W = x.shape
    C_out, C_in, kernel_h, kernel_w = weights.shape

    cols = im2col(x, kernel_h, kernel_w, stride, padding)

    weights_flat = weights.reshape(C_out, -1)

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    output_flat = cols @ weights_flat.T + bias.reshape(1, -1)

    output = output_flat.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    cache = {
        'x_shape': x.shape,
        'weights': weights,
        'cols': cols,
        'stride': stride,
        'padding': padding,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
    }

    return output, cache

# ── Step 018  conv2d_grad_input ──
def conv2d_grad_input(d_out, cache):
    x_shape = cache['x_shape']
    weights = cache['weights']
    stride = cache['stride']
    padding = cache['padding']
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']

    N, C_in, H, W = x_shape
    C_out, C_in_k, kH, kW = weights.shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    weights_flat = weights.reshape(C_out, -1)
    d_cols = d_out_flat @ weights_flat

    dx = col2im(d_cols, x_shape, kernel_h, kernel_w, stride, padding)
    return dx

# ── Step 019  conv2d_grad_weights ──
def conv2d_grad_weights(d_out, cache):
    cols = cache['cols']
    weights_shape = cache['weights'].shape
    C_out, C_in, kH, kW = weights_shape

    N, C_out, out_h, out_w = d_out.shape
    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    dW_flat = cols.T @ d_out_flat

    dW = dW_flat.T.reshape(C_out, C_in, kH, kW)
    return dW

# ── Step 020  conv2d_grad_bias ──
def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

# ── Step 021  conv2d_backward ──
def conv2d_backward(d_out, cache):
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    return dx, dW, db

# ── Step 022  maxpool2d_forward ──
def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w))
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)

    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + kernel
                    w_start = w * stride
                    w_end = w_start + kernel

                    window = x[n, c, h_start:h_end, w_start:w_end]

                    max_val = np.max(window)
                    flat_idx = np.argmax(window)

                    out[n, c, h, w] = max_val
                    argmax[n, c, h, w] = flat_idx

    cache = {
        'x_shape': x.shape,
        'argmax': argmax,
        'kernel': kernel,
        'stride': stride,
    }

    return out, cache

# ── Step 023  scatter_grad_window ──
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    window = np.zeros((kernel, kernel))
    row = argmax_index // kernel
    col = argmax_index % kernel
    window[row, col] = grad_value
    return window

# ── Step 024  maxpool2d_backward ──
def maxpool2d_backward(d_out, cache):
    x_shape = cache['x_shape']
    argmax = cache['argmax']
    kernel = cache['kernel']
    stride = cache['stride']

    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape

    dx = np.zeros(x_shape)

    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    grad_val = d_out[n, c, h, w]
                    argmax_idx = argmax[n, c, h, w]
                    window_grad = scatter_grad_window(grad_val, argmax_idx, kernel)
                    h_start = h * stride
                    w_start = w * stride
                    dx[n, c, h_start:h_start+kernel, w_start:w_start+kernel] += window_grad

    return dx

# ── Step 025  relu_forward ──
def relu_forward(x):
    y = np.maximum(0, x)
    cache = {'x': x}
    return y, cache

# ── Step 026  relu_backward ──
def relu_backward(d_out, cache):
    x = cache['x']
    return d_out * (x > 0)

# ── Step 027  flatten_forward ──
def flatten_forward(x):
    N, C, H, W = x.shape
    out = x.reshape(N, -1)
    cache = {'x_shape': x.shape}
    return out, cache

# ── Step 028  flatten_backward ──
import numpy as np

def flatten_backward(d_out, cache):
    x_shape = cache['x_shape']
    return d_out.reshape(x_shape)

# ── Step 029  linear_forward ──
def linear_forward(x, weights, bias):
    out = x @ weights + bias
    cache = {'x': x, 'weights': weights}
    return out, cache

# ── Step 030  linear_grad_input ──
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    weights = cache['weights']
    dx = d_out @ weights.T
    return dx

# ── Step 031  linear_grad_weights ──
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    return x.T @ dout

# ── Step 032  linear_grad_bias ──
import numpy as np

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

# ── Step 033  linear_backward ──
def linear_backward(dout, cache):
    x = cache['x']
    weights = cache['weights']

    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(x, dout)
    db = linear_grad_bias(dout)

    return dx, dW, db

# ── Step 034  softmax_cross_entropy_forward ──
def softmax_cross_entropy_forward(logits, y, eps=1e-12):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y, eps)
    return loss if loss != -0.0 else 0.0

# ── Step 035  softmax_cross_entropy_backward ──
def softmax_cross_entropy_backward(logits, y):
    probs = stable_softmax(logits)
    N, C = logits.shape
    y_one_hot = one_hot(y, C)
    dlogits = (probs - y_one_hot) / N
    return dlogits

# ── Step 036  sgd_step ──
import numpy as np

def sgd_step(param, grad, lr):
    return param - lr * grad

# ── Step 037  adam_update_m ──
import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one * m + (1 - beta_one) * grad

# ── Step 038  adam_update_v ──
import numpy as np

def adam_update_v(v, grad, beta_two):
    return beta_two * v + (1 - beta_two) * (grad ** 2)

# ── Step 039  adam_bias_correct ──
def adam_bias_correct(moment, beta, t):
    return moment / (1 - beta ** t)

# ── Step 040  adam_param_step ──
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# ── Step 041  adam_step ──
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)

    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)

    return new_param, new_m, new_v

# ── Step 042  init_conv_layer ──
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    fan_in = in_channels * kernel_size * kernel_size
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    W = he_init(shape, fan_in, seed)
    b = init_zero_bias(out_channels)
    return {'W': W, 'b': b}

# ── Step 043  init_linear_layer ──
def init_linear_layer(in_features, out_features, seed=0):
    W = he_init((in_features, out_features), in_features, seed)
    b = init_zero_bias(out_features)
    return {'W': W, 'b': b}

# ── Step 044  init_lenet ──
def init_lenet(in_channels, num_classes, seed=0):
    conv1 = init_conv_layer(
        out_channels=6,
        in_channels=in_channels,
        kernel_size=5,
        seed=seed
    )
    conv2 = init_conv_layer(
        out_channels=16,
        in_channels=6,
        kernel_size=5,
        seed=seed + 1
    )

    flattened_size = 16 * 4 * 4

    fc1 = init_linear_layer(flattened_size, 120, seed = seed + 2)
    fc2 = init_linear_layer(120, num_classes, seed = seed + 3)

    return {
        'conv1': conv1,
        'conv2': conv2,
        'fc1': fc1,
        'fc2': fc2,
    }

# ── Step 045  forward_conv_block ──
def forward_conv_block(x, W, b, pool_size, stride, pad):
    conv_out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = maxpool2d_forward(relu_out, pool_size, pool_size)
    cache = {
        'conv_cache': conv_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache,
    }
    return pool_out, cache

# ── Step 046  forward_classifier_block ──
def forward_classifier_block(x, fc1, fc2):
    flat_out, flatten_cache = flatten_forward(x)
    fc1_out, fc1_cache = linear_forward(flat_out, fc1['W'], fc1['b'])
    relu_out, relu_cache = relu_forward(fc1_out)
    logits, fc2_cache = linear_forward(relu_out, fc2['W'], fc2['b'])
    cache = {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache,
    }
    return logits, cache

# ── Step 047  lenet_forward ──
def lenet_forward(x, params):
    block1_out, block1_cache = forward_conv_block(
        x,
        params['conv1']['W'],
        params['conv1']['b'],
        pool_size=2,
        stride=1,
        pad=0,
    )
    block2_out, block2_cache = forward_conv_block(
        block1_out,
        params['conv2']['W'],
        params['conv2']['b'],
        pool_size=2,
        stride=1,
        pad=0,
    )
    logits, classifier_cache = forward_classifier_block(
        block2_out,
        params['fc1'],
        params['fc2'],
    )
    caches = {
        'block1': block1_cache,
        'block2': block2_cache,
        'classifier': classifier_cache,
    }
    return logits, caches

# ── Step 048  backward_conv_block ──
def backward_conv_block(dout, cache):
    pool_cache = cache['pool_cache']
    relu_cache = cache['relu_cache']
    conv_cache = cache['conv_cache']

    d_relu_out = maxpool2d_backward(dout, pool_cache)
    d_conv_out = relu_backward(d_relu_out, relu_cache)
    dx, dW, db = conv2d_backward(d_conv_out, conv_cache)

    return dx, dW, db

# ── Step 049  backward_classifier_block ──
def backward_classifier_block(dlogits, cache):
    fc2_cache = cache['fc2_cache']
    relu_cache = cache['relu_cache']
    fc1_cache = cache['fc1_cache']
    flatten_cache = cache['flatten_cache']

    d_relu_out, dW2, db2 = linear_backward(dlogits, fc2_cache)
    d_fc1_out = relu_backward(d_relu_out, relu_cache)
    d_flat_out, dW1, db1 = linear_backward(d_fc1_out, fc1_cache)
    dx = flatten_backward(d_flat_out, flatten_cache)

    return {
        'dx': dx,
        'fc1': {'dW': dW1, 'db': db1},
        'fc2': {'dW': dW2, 'db': db2},   
    }

# ── Step 050  lenet_backward ──
def lenet_backward(dlogits, caches):
    classifier_grads = backward_classifier_block(dlogits, caches['classifier'])
    d_block2, dW2, db2 = backward_conv_block(classifier_grads['dx'], caches['block2'])
    d_block1, dW1, db1 = backward_conv_block(d_block2, caches['block1'])

    return {
        'conv1': {'dW': dW1, 'db': db1},
        'conv2': {'dW': dW2, 'db': db2},
        'fc1': {'dW': classifier_grads['fc1']['dW'], 'db': classifier_grads['fc1']['db']},
        'fc2': {'dW': classifier_grads['fc2']['dW'], 'db': classifier_grads['fc2']['db']},
    }

# ── Step 051  lenet_predict ──
def lenet_predict(x, params):
    logits, _ = lenet_forward(x, params)
    return np.argmax(logits, axis=1)

# ── Step 052  build_synthetic_image_dataset ──
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=num_samples)
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    shift = (num_classes - 1) / 2
    for k in range(num_classes):
        mask = y == k
        if np.any(mask):
            x[mask] = x[mask] + (k - shift)

    return x, y

# ── Step 053  shuffle_indices ──
import numpy as np

def shuffle_indices(n, seed=0):
    np.random.seed(seed)
    return np.random.permutation(n)

# ── Step 054  train_test_split ──
def train_test_split(x, y, test_fraction=0.2, seed=0):
    N = len(x)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(N)
    split_idx = int(N * test_fraction)
    train_indices = shuffled_indices[split_idx:]
    test_indices = shuffled_indices[:split_idx]

    tr_f = x[train_indices]
    tr_l = y[train_indices]
    te_f = x[test_indices]
    te_l = y[test_indices]

    return tr_f, tr_l, te_f, te_l

# ── Step 055  iterate_minibatches ──
def iterate_minibatches(x, y, batch_size, seed=0):
    N = x.shape[0]
    idx = shuffle_indices(N, seed)
    
    for start in range(0, N, batch_size):
        if start + batch_size > N:
            break
        end = start + batch_size
        batch_idx = idx[start:end]
        yield x[batch_idx], y[batch_idx]

# ── Step 056  train_step ──
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    logits, caches = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dlogits = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dlogits, caches)

    new_params = {}
    new_opt_state = {}
    
    for layer_name in params:
        new_params[layer_name] = {}
        new_opt_state[layer_name] = {}
        grad_mapping = {'W': 'dW', 'b': 'db'}
        
        for param_name in params[layer_name]:
            param = params[layer_name][param_name]
            grad_name = grad_mapping[param_name]
            grad = grads[layer_name][grad_name]
            
            m = opt_state[layer_name][param_name]['m']
            v = opt_state[layer_name][param_name]['v']
            
            new_param, new_m, new_v = adam_step(
                param, grad, m, v, step, lr, beta_one, beta_two, eps
            )
            
            new_params[layer_name][param_name] = new_param
            new_opt_state[layer_name][param_name] = {'m': new_m, 'v': new_v}
    
    return new_params, new_opt_state, float(loss)

# ── Step 057  train_one_epoch ──
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    losses = []
    current_params = params
    current_opt_state = opt_state
    current_step = step_counter

    for xb, yb in iterate_minibatches(x, y, batch_size, seed):
        current_step += 1
        current_params, current_opt_state, loss = train_step(
            current_params, current_opt_state, xb, yb,
            lr, beta_one, beta_two, eps, current_step
        )
        losses.append(loss)

    return current_params, current_opt_state, current_step, losses

# ── Step 058  train_loop ──
def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3, beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    opt_state = {}
    for layer_name in params:
        opt_state[layer_name] = {}
        for param_name in params[layer_name]:
            param = params[layer_name][param_name]
            opt_state[layer_name][param_name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }
    
    current_params = params
    current_opt_state = opt_state
    step_counter = 0
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_seed = seed + epoch
        current_params, current_opt_state, step_counter, losses = train_one_epoch(
            current_params, current_opt_state, x_train, y_train, batch_size,
            lr, beta_one, beta_two, eps, step_counter, epoch_seed
        )
        loss_history.extend(losses)
    
    return current_params, loss_history

# ── Step 059  evaluate ──
def evaluate(params, x, y):
    preds = lenet_predict(x, params)
    return np.mean(preds == y)

# ── Scaffold (runner) ──
"""Scaffold demo for a NumPy LeNet-style CNN trained end-to-end."""

import numpy as np

from solution import (
    argmax_rows,
    row_max,
    row_sum,
    exp_shifted,
    stable_softmax,
    one_hot,
    gather_true_class_probs,
    cross_entropy_loss,
    accuracy,
    he_std,
    he_init,
    init_zero_bias,
    pad_2d,
    output_spatial_size,
    im2col,
    col2im,
    conv2d_forward,
    conv2d_grad_input,
    conv2d_grad_weights,
    conv2d_grad_bias,
    conv2d_backward,
    maxpool2d_forward,
    scatter_grad_window,
    maxpool2d_backward,
    relu_forward,
    relu_backward,
    flatten_forward,
    flatten_backward,
    linear_forward,
    linear_grad_input,
    linear_grad_weights,
    linear_grad_bias,
    linear_backward,
    softmax_cross_entropy_forward,
    softmax_cross_entropy_backward,
    sgd_step,
    adam_update_m,
    adam_update_v,
    adam_bias_correct,
    adam_param_step,
    adam_step,
    init_conv_layer,
    init_linear_layer,
    init_lenet,
    forward_conv_block,
    forward_classifier_block,
    lenet_forward,
    backward_conv_block,
    backward_classifier_block,
    lenet_backward,
    lenet_predict,
    build_synthetic_image_dataset,
    shuffle_indices,
    train_test_split,
    iterate_minibatches,
    train_step,
    train_one_epoch,
    train_loop,
    evaluate,
)


def _print_param_shapes(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _print_param_shapes(v, prefix=f"{prefix}{k}.")
    elif isinstance(obj, np.ndarray):
        print(f"  {prefix[:-1]}: shape={obj.shape}")
    else:
        print(f"  {prefix[:-1]}: {type(obj).__name__}")


if __name__ == "__main__":
    np.random.seed(0)

    # ---- Dataset ---------------------------------------------------------
    num_samples = 64
    num_classes = 3
    image_size = 28
    in_channels = 1

    x, y = build_synthetic_image_dataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        in_channels=in_channels,
        seed=0,
    )
    print(f"Dataset shapes: x={x.shape}, y={y.shape}")
    print(f"Label distribution: {np.bincount(y, minlength=num_classes)}")

    x_train, y_train, x_test, y_test = train_test_split(x, y, test_fraction=0.25, seed=0)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # ---- Model init ------------------------------------------------------
    params = init_lenet(in_channels=in_channels, num_classes=num_classes, seed=0)
    print("Parameter tensors:")
    _print_param_shapes(params)

    # ---- One forward pass before training -------------------------------
    logits0, _ = lenet_forward(x_train[:8], params)
    probs0 = stable_softmax(logits0)
    init_loss = cross_entropy_loss(probs0, y_train[:8])
    init_acc = accuracy(logits0, y_train[:8])
    print(f"Initial mini-batch loss: {init_loss:.4f}, accuracy: {init_acc:.3f}")

    # ---- Train -----------------------------------------------------------
    params, loss_history = train_loop(
        params,
        x_train,
        y_train,
        num_epochs=3,
        batch_size=16,
        lr=1e-3,
        beta_one=0.9,
        beta_two=0.999,
        eps=1e-8,
        seed=0,
    )
    print(f"Training steps: {len(loss_history)}")
    print(f"First loss: {loss_history[0]:.4f}, last loss: {loss_history[-1]:.4f}")

    # ---- Evaluate --------------------------------------------------------
    train_acc = evaluate(params, x_train, y_train)
    test_acc = evaluate(params, x_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    # ---- Sample predictions ---------------------------------------------
    preds = lenet_predict(x_test[:8], params)
    print(f"Sample predictions: {preds.tolist()}")
    print(f"Sample labels:      {y_test[:8].tolist()}")
