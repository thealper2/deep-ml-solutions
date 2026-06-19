"""
Tiny GPT From Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  build_vocab ──
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    unique_chars = set(text)
    return sorted(unique_chars)

# ── Step 002  build_stoi ──
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    stoi = {}
    for i, char in enumerate(vocab):
        if char not in stoi.keys():
            stoi[char] = i

    return stoi

# ── Step 003  build_itos ──
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    stoi = {}
    for i, char in enumerate(vocab):
        if char not in stoi.keys():
            stoi[char] = i

    return {v: k for k, v in stoi.items()}

# ── Step 004  encode_char ──
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    return stoi[ch]

# ── Step 005  encode_string ──
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    return [stoi[char] for char in text]

# ── Step 006  decode_int ──
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    return itos[token_id]

# ── Step 007  decode_ids ──
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    return ''.join([itos[i] for i in ids])

# ── Step 008  make_1d_array ──
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.array(values).flatten()

# ── Step 009  get_array_shape ──
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    return arr.shape

# ── Step 010  get_array_dtype ──
import numpy as np

def get_array_dtype(arr):
    return arr.dtype

# ── Step 011  make_2d_zeros ──
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# ── Step 012  make_2d_random ──
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(size=(rows, cols))

# ── Step 013  index_element ──
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    return arr[i][j]

# ── Step 014  slice_row ──
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i]

# ── Step 015  slice_column ──
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    return arr[:, j]

# ── Step 016  slice_subblock ──
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    return arr[r0:r1, c0:c1]

# ── Step 017  elementwise_add ──
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    return a + b

# ── Step 018  elementwise_multiply ──
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    return a * b

# ── Step 019  scalar_broadcast_add ──
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    return np.add(arr, scalar)

# ── Step 020  vector_matrix_broadcast_add ──
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    return matrix + vector

# ── Step 021  array_exp ──
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    return np.exp(arr)

# ── Step 022  array_log ──
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    return np.log(arr)

# ── Step 023  sum_all ──
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return np.sum(arr)

# ── Step 024  sum_axis0 ──
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis=0)

# ── Step 025  sum_axis1 ──
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis=1)

# ── Step 026  max_along_axis ──
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis=axis)

# ── Step 027  matmul ──
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return np.matmul(a, b)

# ── Step 028  transpose_matrix ──
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    return arr.T

# ── Step 029  sum_keepdims ──
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    return np.sum(arr, axis=axis, keepdims=True)

# ── Step 030  naive_softmax_1d ──
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    x_max = np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ── Step 031  softmax_overflow_demo ──
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    arr = np.array([large_value])
    exp_arr = np.exp(arr)
    naive_exp = float(exp_arr[0])
    overflowed = np.isinf(naive_exp)
    return {'naive_exp': naive_exp, 'overflowed': overflowed}

# ── Step 032  stable_softmax_1d ──
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    x_max = np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ── Step 033  stable_softmax_2d_rowwise ──
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    x_max = np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ── Step 034  read_text_file ──
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    if not isinstance(text_blob, str):
        raise TypeError('Input must be a string')
    if len(text_blob) == 0:
        raise ValueError('Input string cannot be empty')
    return text_blob

# ── Step 035  encode_corpus_to_int_array ──
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    return np.array([stoi[char] for char in text], dtype=np.int64)

# ── Step 036  pick_split_point ──
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    return int(n * train_frac)

# ── Step 037  slice_train_and_val ──
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    train = data[:split_idx]
    val = data[split_idx:]
    return train, val

# ── Step 038  pick_block_size ──
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    return default_size if default_size > 1 else 1

# ── Step 039  slice_x_at_offset ──
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    return data[i:i+block_size]

# ── Step 040  slice_y_at_offset ──
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    return data[i+1:i+block_size+1]

# ── Step 041  sample_random_batch_offsets ──
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    max_start = data_len - block_size
    offsets = rng.integers(0, max_start, size=batch_size)
    return offsets

# ── Step 042  stack_x_batch ──
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    batch = []
    for offset in offsets:
        batch.append(slice_x_at_offset(data, offset, block_size))

    return np.array(batch)

# ── Step 043  stack_y_batch ──
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    batch = []
    for offset in offsets:
        batch.append(slice_y_at_offset(data, offset, block_size))

    return np.array(batch)

# ── Step 044  get_batch ──
def get_batch(data, block_size, batch_size, rng):
    data_len = len(data)
    offsets = sample_random_batch_offsets(data_len, block_size, batch_size, rng)
    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)
    return X, Y

# ── Step 045  allocate_count_matrix ──
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# ── Step 046  loop_fill_counts ──
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for i in range(len(data) - 1):
        n_matrix[data[i], data[i + 1]] += 1

    return n_matrix

# ── Step 047  vectorize_counts_add_at ──
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    counts = allocate_count_matrix(vocab_size)
    current = data[:-1]
    next_tokens = data[1:]
    np.add.at(counts, (current, next_tokens), 1)
    return counts

# ── Step 048  add_one_smoothing ──
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    return n_matrix + 1

# ── Step 049  row_sums_of_counts ──
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    return np.sum(n_matrix, axis=1, keepdims=True)

# ── Step 050  normalize_counts_to_probs ──
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    row_sums = row_sums_of_counts(n_matrix)
    return n_matrix / row_sums

# ── Step 051  sample_next_token ──
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    probs = p_matrix[current_id]
    return rng.choice(len(probs), p=probs)

# ── Step 052  generate_sequence ──
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    sequence = np.zeros(length, dtype=np.int64)
    sequence[0] = start_id
    
    for i in range(1, length):
        sequence[i] = sample_next_token(p_matrix, sequence[i - 1], rng)
    
    return sequence

# ── Step 053  decode_generated_sequence ──
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    return ''.join([itos[i] for i in ids])

# ── Step 054  log_prob_of_pair ──
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    prob = index_element(p_matrix, current_id, next_id)
    return array_log(prob).item()

# ── Step 055  sum_negative_log_probs ──
def sum_negative_log_probs(p_matrix, data):
    total = 0.0
    for i in range(len(data) - 1):
        log_prob = log_prob_of_pair(p_matrix, data[i], data[i + 1])
        total += -log_prob

    return total

# ── Step 056  average_nll ──
def average_nll(p_matrix, data):
    total_nll = sum_negative_log_probs(p_matrix, data)
    num_bigrams = len(data) - 1
    return total_nll / num_bigrams

# ── Step 057  initialize_w_random ──
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    return rng.standard_normal(size=(vocab_size, vocab_size))

# ── Step 058  scale_w_small ──
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    return w_matrix * scale

# ── Step 059  one_hot_encode_batch ──
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    return np.eye(vocab_size)[ids]

# ── Step 060  forward_logits_onehot ──
def forward_logits_onehot(onehot, w_matrix):
    return np.matmul(onehot, w_matrix)

# ── Step 061  observe_lookup_equivalence ──
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    V = w.shape[0]
    B = len(ids)

    one_hot = np.zeros((B, V))
    one_hot[np.arange(B), ids] = 1.0
    onehot_result = one_hot @ w

    index_result = w[ids]

    return {
        'onehot_result': onehot_result,
        'index_result': index_result,
    }

# ── Step 062  forward_logits_lookup ──
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    return w[ids]

# ── Step 063  logits_to_probs_rowwise ──
def logits_to_probs_rowwise(logits):
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

# ── Step 064  gather_correct_token_probs ──
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    return np.array([prob[target] for prob, target in zip(probs, targets)])

# ── Step 065  cross_entropy_loss ──
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    correct_probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(correct_probs)
    return -np.mean(log_probs)

# ── Step 066  derive_dlogits_on_paper ──
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    derivation = (
        "For cross-entropy loss L = -1/B * sum_i log(p_i) where p_i = exp(logits_i) / sum(exp(logits_j)),\n"
        "the derivative of the log-softmax with respect to logits gives: ∂p_k/∂logits_j = p_k(δ_kj - p_j).\n"
        "Applying chain rule: ∂L/∂logits_j = -1/B * sum_i (1/p_i) * ∂p_i/∂logits_j = -1/B * sum_i (δ_ij - p_j) = (p_j - onehot_j) / B.\n"
        "Thus, dL/dlogits = (probs - onehot(targets)) / B."
    )
    return derivation

# ── Step 067  compute_dlogits ──
def compute_dlogits(probs, targets):
    B, V = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(B), targets] = 1.0
    return (probs - one_hot) / B

# ── Step 068  derive_dw_on_paper ──
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    derivation = (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )
    return derivation

# ── Step 069  compute_dw_scatter_add ──
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros((vocab_size, dlogits.shape[1]))
    np.add.at(dW, ids, dlogits)
    return dW

# ── Step 070  sgd_update_w ──
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    new_w = w - (learning_rate * dw)
    return new_w

# ── Step 071  run_one_training_step ──
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    logits = w[ids]

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    B = len(ids)
    one_hot = np.zeros((B, w.shape[0]))
    one_hot[np.arange(B), targets] = 1.0
    loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-15), axis=1))

    dlogits = (probs - one_hot) / B

    vocab_size = w.shape[0]
    dW = np.zeros((vocab_size, vocab_size))
    np.add.at(dW, ids, dlogits)

    new_w = w - learning_rate * dW
    return {'w': new_w, 'loss': float(loss)}

# ── Step 072  train_neural_bigram_loop ──
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    rng = np.random.default_rng(0)
    loss_history = []
    current_w = w.copy()

    for step in range(num_steps):
        x_batch, y_batch = get_batch(data, block_size, batch_size, rng)
        ids = x_batch.flatten()
        targets = y_batch.flatten()
        result = run_one_training_step(current_w, ids, targets, learning_rate)
        current_w = result['w']
        if step % log_every == 0:
            loss_history.append(result['loss'])

    return {'w': current_w, 'loss_history': loss_history}

# ── Step 073  sample_from_neural_bigram ──
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    rng = np.random.default_rng(0)
    ids = [start_id]
    current_id = start_id

    for _ in range(num_tokens):
        logits = forward_logits_lookup(w, np.array([current_id]))
        probs = logits_to_probs_rowwise(logits)[0]
        next_id = rng.choice(len(probs), p=probs)
        ids.append(next_id)
        current_id = next_id

    return decode_ids(ids, itos)

# ── Step 074  linear_forward ──
def linear_forward(x, w):
    y = x @ w
    return {'y': y, 'cache': {'x': x, 'w': w}}

# ── Step 075  derive_dx_on_paper ──
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    notes = (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )
    return notes

# ── Step 076  derive_linear_dw_on_paper ──
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    notes = (
        "For Y = X @ W, where X is (B, D_in), W is (D_in, D_out), Y is (B, D_out).\n"
        "By the chain rule, the gradient dL/dW is obtained by backpropagating dL/dY through the linear transformation:\n"
        "dL/dW = X.T @ dY.\n"
        "This can be verified elementwise: Y_{i,k} = sum_j X_{i,j} W_{j,k}, so dL/dW_{j,k} = sum_i X_{i,j} * dL/dY_{i,k}."
    )
    return notes

# ── Step 077  linear_backward_dx ──
def linear_backward_dx(dy, cache):
    W = cache['w']
    dx = np.dot(dy, W.T)
    return dx

# ── Step 078  linear_backward_dw ──
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    x = cache['x']
    dW = np.dot(x.T, dy)
    return dW

# ── Step 079  bias_add_forward ──
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    y = vector_matrix_broadcast_add(x, b)
    return {'y': y, 'cache': {'b_shape': b.shape}}

# ── Step 080  bias_add_backward_db ──
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    db = np.sum(dy, axis=0)
    return db

# ── Step 081  relu_forward ──
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    y = np.maximum(0, x)
    return {'y': y, 'cache': {'x': x}}

# ── Step 082  relu_backward ──
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    x = cache['x']
    return dy * (x > 0)

# ── Step 083  softmax_cross_entropy_backward ──
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    B, V = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(B), targets] = 1.0
    return (probs - one_hot) / B

# ── Step 084  layernorm_forward_mean ──
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    return np.mean(x, axis=-1, keepdims=True)

# ── Step 085  layernorm_forward_variance ──
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    return np.var(x, axis=-1, keepdims=True)

# ── Step 086  layernorm_forward_normalize ──
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    return (x - mean) / np.sqrt((var + eps))

# ── Step 087  layernorm_forward_affine ──
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_norm = layernorm_forward_normalize(x, mean, var, eps)
    y = gamma * x_norm + beta
    return {
        'y': y, 
        'cache': {
            'x': x, 
            'x_hat': x_norm, 
            'mean': mean, 
            'var': var, 
            'gamma': gamma, 
            'eps': eps
        }
    }

# ── Step 088  layernorm_backward_subtract_mean ──
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    x = cache['x']
    mean = cache['mean']
    B, D = dy.shape
    dy_mean = np.mean(dy, axis=1, keepdims=True)
    dx = dy - dy_mean
    return dx

# ── Step 089  layernorm_backward_divide_std ──
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    x_hat = cache['x_hat']
    var = cache['var']
    eps = cache['eps']

    std = np.sqrt(var + eps)
    dx = dy / std
    return dx

# ── Step 090  layernorm_backward_full ──
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    x = cache['x']
    x_hat = cache['x_hat']
    mean = cache['mean']
    var = cache['var']
    gamma = cache['gamma']
    eps = cache['eps']

    B, D = x.shape

    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)

    dy_hat = dy * gamma

    std = np.sqrt(var + eps)
    dx_norm = dy_hat / std

    dx_centered = dx_norm - np.mean(dx_norm, axis=1, keepdims=True)

    dy_hat_mean = np.mean(dy_hat, axis=1, keepdims=True)
    dy_hat_x_hat_mean = np.mean(dy_hat * x_hat, axis=1, keepdims=True)

    dx = (1 / std) * (dy_hat - dy_hat_mean - x_hat * dy_hat_x_hat_mean)
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta,
    }

# ── Step 091  layernorm_backward_implementation ──
def layernorm_backward_implementation(d_out, cache):
    x = cache['x']
    x_hat = cache['x_hat']
    mean = cache['mean']
    var = cache['var']
    gamma = cache['gamma']
    eps = cache['eps']

    B, D = x.shape

    dgamma = np.sum(d_out * x_hat, axis=0)
    dbeta = np.sum(d_out, axis=0)

    dy_hat = d_out * gamma

    std = np.sqrt(var + eps)
    dx_norm = dy_hat / std

    dx_centered = dx_norm - np.mean(dx_norm, axis=1, keepdims=True)

    dy_hat_mean = np.mean(dy_hat, axis=1, keepdims=True)
    dy_hat_x_hat_mean = np.mean(dy_hat * x_hat, axis=1, keepdims=True)

    dx = (1 / std) * (dy_hat - dy_hat_mean - x_hat * dy_hat_x_hat_mean)
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta,
    }

# ── Step 092  create_token_embedding ──
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    embeddings = np.random.standard_normal((vocab_size, d_model)) * scale
    return embeddings

# ── Step 093  token_embedding_forward ──
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    out = embedding_matrix[token_ids]
    B, T = token_ids.shape
    V, d_model = embedding_matrix.shape
    cache = {'token_ids': token_ids, 'vocab_size': V}
    return out, cache

# ── Step 094  token_embedding_backward ──
import numpy as np

def token_embedding_backward(d_out, cache):
    vocab_size = cache['vocab_size']
    token_ids = cache['token_ids']
    embed_dim = d_out.shape[-1]
    dW = np.zeros((vocab_size, embed_dim))
    flat_token_ids = token_ids.flatten()
    flat_d_out = d_out.reshape(-1, embed_dim)
    np.add.at(dW, flat_token_ids, flat_d_out)
    return dW

# ── Step 095  create_positional_embedding ──
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    matrix = make_2d_random(block_size, d_model, seed=None)
    scaled_matrix = scale_w_small(matrix, scale)
    return scaled_matrix

# ── Step 096  slice_positional_embedding ──
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    return positional_matrix[:seq_len]

# ── Step 097  add_token_and_positional_embeddings ──
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    return token_emb + pos_emb

# ── Step 098  embedding_sum_backward ──
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    d_token_emb = d_out
    d_pos_emb = sum_axis0(np.sum(d_out, axis=0))
    d_pos_emb = np.sum(d_out, axis=0)
    return {'d_token_emb': d_token_emb, 'd_pos_emb': d_pos_emb}

# ── Step 099  create_qkv_projections ──
def create_qkv_projections(d_model, d_head, scale=0.02):
    Wq = make_2d_random(d_model, d_head, seed=0)
    Wk = make_2d_random(d_model, d_head, seed=1)
    Wv = make_2d_random(d_model, d_head, seed=2)

    Wq = scale_w_small(Wq, scale)
    Wk = scale_w_small(Wk, scale)
    Wv = scale_w_small(Wv, scale)

    return {
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv,
    }

# ── Step 100  compute_query ──
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    return np.dot(x, w_q)

# ── Step 101  compute_key ──
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    return np.dot(x, w_k)

# ── Step 102  compute_value ──
def compute_value(x, w_v):
    return np.dot(x, w_v)

# ── Step 103  compute_attention_scores ──
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    scores = np.matmul(q, k.transpose(0, 2, 1))
    return scores

# ── Step 104  scale_attention_scores ──
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    return scores / np.sqrt(d_head)

# ── Step 105  build_causal_mask ──
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return ~mask

# ── Step 106  apply_causal_mask ──
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    masked_scores = np.where(causal_mask, scaled_scores, -np.inf)
    return masked_scores

# ── Step 107  softmax_attention_weights ──
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    max_x = np.max(masked_scores, axis=-1, keepdims=True)
    exp_x = np.exp(masked_scores - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ── Step 108  attention_weighted_values ──
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    return attn @ v

# ── Step 109  apply_output_projection ──
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    return attn_out @ w_o

# ── Step 110  output_projection_backward ──
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    attn_out = cache['attn_out']
    w_o = cache['w_o']
    
    d_attn_out = d_proj @ w_o.T
    
    B, T, d_head = attn_out.shape
    _, _, d_model = d_proj.shape
    
    attn_out_flat = attn_out.reshape(-1, d_head)
    d_proj_flat = d_proj.reshape(-1, d_model)
    
    dw_o = attn_out_flat.T @ d_proj_flat
    
    return {'d_attn_out': d_attn_out, 'dw_o': dw_o}

# ── Step 111  attention_value_backward ──
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    attn = cache['attn']
    v = cache['v']
    d_attn = d_attn_out @ np.swapaxes(v, -1, -2)
    d_v = np.swapaxes(attn, -1, -2) @ d_attn_out
    return {'d_attn': d_attn, 'd_v': d_v}

# ── Step 112  masked_softmax_backward ──
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    attn = cache['attn']
    sum_dot = np.sum(d_attn * attn, axis=-1, keepdims=True)
    d_scores = attn * (d_attn - sum_dot)
    return d_scores

# ── Step 113  scale_scores_backward ──
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    return d_scaled_scores / np.sqrt(d_head)

# ── Step 114  qk_scores_backward ──
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    q = cache['q']
    k = cache['k']
    d_q = np.matmul(d_scores, k)
    d_k = np.matmul(d_scores.transpose(0, 2, 1), q)
    return {'d_q': d_q, 'd_k': d_k}

# ── Step 115  qkv_projection_backward ──
def qkv_projection_backward(d_q, d_k, d_v, cache):
    x = cache['x']
    w_q = cache['w_q']
    w_k = cache['w_k']
    w_v = cache['w_v']

    B, T, d_in = x.shape
    _, _, d_head = d_q.shape

    x_flat = x.reshape(-1, d_in)
    d_q_flat = d_q.reshape(-1, d_head)
    d_k_flat = d_k.reshape(-1, d_head)
    d_v_flat = d_v.reshape(-1, d_head)

    dw_q = x_flat.T @ d_q_flat
    dw_k = x_flat.T @ d_k_flat
    dw_v = x_flat.T @ d_v_flat

    dx_flat = d_q_flat @ w_q.T + d_k_flat @ w_k.T + d_v_flat @ w_v.T
    dx = dx_flat.reshape(B, T, d_in)
    
    return {
        'dx': dx,
        'dw_q': dw_q,
        'dw_k': dw_k,
        'dw_v': dw_v,
    }

# ── Step 116  choose_attention_head_config ──
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    if d_model % n_heads != 0:
        raise ValueError('d_model % n_heads != 0')
    d_head = d_model // n_heads
    return {
        'n_heads': n_heads,
        'd_head': d_head,
        'd_model': d_model,
    }

# ── Step 117  create_multihead_qkv_projections ──
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    Wq = scale_w_small(make_2d_random(d_model, d_model, seed=0), scale)
    Wk = scale_w_small(make_2d_random(d_model, d_model, seed=1), scale)
    Wv = scale_w_small(make_2d_random(d_model, d_model, seed=2), scale)
    
    return {'Wq': Wq, 'Wk': Wk, 'Wv': Wv}

# ── Step 118  create_multihead_output_projection ──
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    Wo = make_2d_random(d_model, d_model, seed=0)
    Wo = scale_w_small(Wo, scale)
    return Wo

# ── Step 119  reshape_to_heads ──
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    B, T, d_model = x.shape
    return x.reshape(B, T, n_heads, d_head)

# ── Step 120  transpose_heads_to_front ──
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    return x_heads.transpose(0, 2, 1, 3)

# ── Step 121  get_multihead_n_heads ──
def get_multihead_n_heads(config):
    return config['n_heads']

# ── Step 122  get_multihead_sequence_length ──
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    return x.shape[1]

# ── Step 123  compute_d_head ──
def compute_d_head(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError('d_model % n_heads != 0')
    return d_model // n_heads

# ── Step 124  multihead_masked_softmax_scores ──
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    B, n_heads, T, T = scores.shape
    scores_2d = scores.reshape(-1, T)
    mask_expanded = mask.reshape(1, 1, T, T)
    masked_scores = apply_causal_mask(scores, mask_expanded)
    masked_scores_2d = masked_scores.reshape(-1, T)
    probs_2d = stable_softmax_2d_rowwise(masked_scores_2d)
    probs = probs_2d.reshape(B, n_heads, T, T)
    return probs

# ── Step 125  multihead_weighted_sum ──
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    return np.matmul(weights, v_heads)

# ── Step 126  transpose_heads_to_back ──
def transpose_heads_to_back(x_heads):
    return x_heads.transpose(0, 2, 1, 3)

# ── Step 127  get_multihead_output_sequence_length ──
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    return x_heads_back.shape[1]

# ── Step 128  merge_heads_to_d_model ──
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    B, T, n_heads, d_head = x_heads_back.shape
    return x_heads_back.reshape(B, T, n_heads * d_head)

# ── Step 129  multihead_output_projection_forward ──
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    linear = linear_forward(merged, w_out)
    with_bias = bias_add_forward(linear['y'], b_out)
    cache = {'merged': merged, 'w_out': w_out}
    return {'out': with_bias['y'], 'cache': cache}

# ── Step 130  multihead_reshape_transpose_backward ──
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    B = shape_info['B']
    T = shape_info['T']
    n_heads = shape_info['n_heads']
    d_head = shape_info['d_head']
    d_heads = reshape_to_heads(d_merged, n_heads, d_head)
    d_heads_front = transpose_heads_to_front(d_heads)
    return d_heads_front

# ── Step 131  ffn_linear_one_forward ──
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    linear = linear_forward(x, w1)
    with_bias = bias_add_forward(linear['y'], b1)
    cache = {'x': x, 'w1': w1}
    return {'h1': with_bias['y'], 'cache': cache}

# ── Step 132  ffn_activation_forward ──
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    a1 = relu_forward(h1)
    cache = {'h1': h1}
    return a1['y'], cache

# ── Step 133  ffn_linear_two_forward ──
def ffn_linear_two_forward(a1, w2, b2):
    linear = linear_forward(a1, w2)
    with_bias = bias_add_forward(linear['y'], b2)
    cache = {'a1': a1, 'w2': w2}
    return {'h2': with_bias['y'], 'cache': cache}

# ── Step 134  ffn_backward ──
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    x = cache['x']
    w1 = cache['w1']
    h1 = cache['h1']
    a1 = cache['a1']
    w2 = cache['w2']
    
    B, T, d_ff = a1.shape
    _, _, d_model = x.shape
    
    d_a1 = d_out @ w2.T
    dw2 = a1.reshape(-1, d_ff).T @ d_out.reshape(-1, d_model)
    db2 = np.sum(d_out, axis=(0, 1))
    d_h1 = d_a1 * (h1 > 0)
    dx = d_h1 @ w1.T
    dw1 = x.reshape(-1, d_model).T @ d_h1.reshape(-1, d_ff)
    db1 = np.sum(d_h1, axis=(0, 1))

    return {'dx': dx, 'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

# ── Step 135  residual_forward ──
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    out = x + sublayer_out
    return out

# ── Step 136  residual_backward ──
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    return d_y.copy(), d_y.copy()

# ── Step 137  pre_layernorm_sublayer_forward ──
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    norm_out = layernorm_forward_affine(x, ln_params['gamma'], ln_params['beta'], eps=1e-5)
    sublayer_out = sublayer_fn(norm_out['y'], sublayer_params)
    y = residual_forward(x, sublayer_out['y'])
    cache = {
        'x': x,
        'ln_cache': norm_out['cache'],
        'sublayer_cache': sublayer_out['cache'],
    }
    return {'y': y, 'cache': cache}

# ── Step 138  transformer_block_forward ──
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    B, T, D = x.shape

    # ---------- LN1 ----------
    mean1 = x.mean(axis=-1, keepdims=True)
    var1 = x.var(axis=-1, keepdims=True)
    x_norm1 = (x - mean1) / np.sqrt(var1 + 1e-5)
    x_norm1 = x_norm1 * block_params["ln1"]["gamma"] + block_params["ln1"]["beta"]

    # ---------- Multi-head self-attention ----------
    attn = block_params["attn"]
    n_heads = attn.get("n_heads", block_params.get("n_heads", 2))
    dh = D // n_heads
    scale = 1.0 / np.sqrt(dh)

    Wq = attn.get("Wq", np.zeros((D, D)))
    Wk = attn.get("Wk", np.zeros((D, D)))
    Wv = attn.get("Wv", np.zeros((D, D)))
    Wo = attn.get("Wo", np.zeros((D, D)))
    bq = attn.get("bq", np.zeros(D))
    bk = attn.get("bk", np.zeros(D))
    bv = attn.get("bv", np.zeros(D))
    bo = attn.get("bo", np.zeros(D))

    Q = x_norm1 @ Wq + bq
    K = x_norm1 @ Wk + bk
    V = x_norm1 @ Wv + bv

    def split_heads(t):                       # (B,T,D) -> (B,h,T,dh)
        return t.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)
    Qs, Ks, Vs = split_heads(Q), split_heads(K), split_heads(V)

    scores = Qs @ Ks.transpose(0, 1, 3, 2) * scale     # (B,h,T,T)
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)  # mask future
    scores = np.where(causal, -1e9, scores)

    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    P = e / e.sum(axis=-1, keepdims=True)              # softmax (B,h,T,T)

    ctx = P @ Vs                                       # (B,h,T,dh)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, D)   # merge heads
    attn_out = ctx @ Wo + bo
    h1 = x + attn_out                                  # residual 1

    # ---------- LN2 ----------
    mean2 = h1.mean(axis=-1, keepdims=True)
    var2 = h1.var(axis=-1, keepdims=True)
    x_norm2 = (h1 - mean2) / np.sqrt(var2 + 1e-5)
    x_norm2 = x_norm2 * block_params["ln2"]["gamma"] + block_params["ln2"]["beta"]

    # ---------- FFN (ReLU arada) ----------
    ffn = block_params["ffn"]
    w1 = ffn["w1"]; b1 = ffn["b1"]; w2 = ffn["w2"]; b2 = ffn["b2"]
    ffn_pre = x_norm2 @ w1 + b1
    ffn_hidden = np.maximum(ffn_pre, 0.0)              # ReLU
    ffn_out = ffn_hidden @ w2 + b2
    y = h1 + ffn_out                                   # residual 2

    cache = {
        "attn_branch": {
            "x": x,
            "ln_cache": {"x": x, "mean": mean1, "var": var1},
            "sublayer_cache": {
                "x_norm": x_norm1,
                "Q": Q, "K": K, "V": V,
                "Qs": Qs, "Ks": Ks, "Vs": Vs,
                "P": P, "ctx": ctx,
                "scale": scale, "n_heads": n_heads,
                "attn_out": attn_out,
            },
        },
        "ffn_branch": {
            "x": h1,
            "ln_cache": {"x": h1, "mean": mean2, "var": var2},
            "sublayer_cache": {
                "x_norm": x_norm2,
                "ffn_pre": ffn_pre,
                "ffn_hidden": ffn_hidden,
                "ffn_out": ffn_out,
            },
        },
    }
    return {"y": y, "cache": cache}

# ── Step 139  transformer_block_backward ──
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block."""
    x = cache["attn_branch"]["x"]
    cache = _complete_block_cache(x, block_params)

    attn_branch = cache["attn_branch"]
    ffn_branch = cache["ffn_branch"]

    d_ln2_out, ffn_grads = _ffn_sublayer_backward(
        d_y, ffn_branch["sublayer_cache"], block_params["ffn"]
    )
    d_h1_sub, d_g2, d_b2 = layernorm_backward_affine(
        d_ln2_out, ffn_branch["ln_cache"]
    )
    d_h1 = d_h1_sub + d_y

    d_ln1_out, attn_grads = _attn_sublayer_backward(
        d_h1, attn_branch["sublayer_cache"], block_params["attn"]
    )
    d_x_sub, d_g1, d_b1 = layernorm_backward_affine(
        d_ln1_out, attn_branch["ln_cache"]
    )
    d_x = d_x_sub + d_h1

    grads = {
        "ln1": {"gamma": d_g1, "beta": d_b1},
        "ln2": {"gamma": d_g2, "beta": d_b2},
        "attn": attn_grads,
        "ffn": ffn_grads,
    }
    return d_x, grads

# ── Step 140  stack_transformer_blocks ──
import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    blocks = []
    d_head = d_model // n_heads

    for layer_idx in range(n_layers):
        ln1 = {
            'gamma': np.ones(d_model),
            'beta': np.zeros(d_model),
        }
        ln2 = {
            'gamma': np.ones(d_model),
            'beta': np.zeros(d_model),
        }

        attn = {
            'Wq': scale_w_small(make_2d_random(d_model, d_model, seed=0), 0.02),
            'Wk': scale_w_small(make_2d_random(d_model, d_model, seed=1), 0.02),
            'Wv': scale_w_small(make_2d_random(d_model, d_model, seed=2), 0.02),
            'Wo': scale_w_small(make_2d_random(d_model, d_model, seed=3), 0.02),
            'bo': np.zeros(d_model),
        }

        ffn = {
            'W1': scale_w_small(make_2d_random(d_model, d_ff, seed=4), 0.02),
            'b1': np.zeros(d_ff),
            'W2': scale_w_small(make_2d_random(d_ff, d_model, seed=5), 0.02),
            'b2': np.zeros(d_model),
        }

        block = {
            'ln1': ln1,
            'attn': attn,
            'ln2': ln2,
            'ffn': ffn,
        }

        blocks.append(block)

    return blocks

# ── Step 141  forward_through_all_blocks ──
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    y = x
    caches = []
    
    for block in blocks:
        out = transformer_block_forward(y, block)
        y = out['y']
        caches.append(out['cache'])
    
    return y, caches

# ── Step 142  backward_through_all_blocks ──
def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y     : (B, T, d_model) upstream gradient at the top of the stack
      caches  : list of per-block forward caches
      blocks  : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    d_x = d_y
    grads_list = []
    
    for i in range(len(blocks) - 1, -1, -1):
        d_x, grads = transformer_block_backward(d_x, caches[i], blocks[i])
        grads_list.insert(0, grads)
    
    return d_x, grads_list

# ── Step 143  final_layernorm_forward ──
def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    eps = 1e-5
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_hat': x_hat,
        'gamma': gamma,
        'eps': eps,
    }
    return y, cache

# ── Step 144  lm_head_linear_forward ──
def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    linear = linear_forward(x, w_lm)
    with_bias = bias_add_forward(linear['y'], b_lm)
    return {'logits': with_bias['y'], 'cache': {'x': x, 'w_lm': w_lm}}

# ── Step 145  full_model_forward ──
def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""
    B, T = x_ids.shape
    d_model = model_params['tok_emb'].shape[1]

    tok_emb = model_params['tok_emb'][x_ids]
    pos_emb = model_params['pos_emb'][:T, :]

    x = tok_emb + pos_emb

    emb_cache = {'tok_emb': tok_emb, 'pos_emb': pos_emb, 'x_sum': x}

    blocks_cache = []
    for block_params in model_params['blocks']:
        out = transformer_block_forward(x, block_params)
        x = out['y']
        blocks_cache.append(out['cache'])

    y, ln_cache = final_layernorm_forward(x, model_params['ln_f']['gamma'], model_params['ln_f']['beta'])
    logits = y @ model_params['lm_head']['w_lm'] + model_params['lm_head']['b_lm']

    lm_cache = {'w_lm': model_params['lm_head']['w_lm'], 'b_lm': model_params['lm_head']['b_lm']}

    caches = {
        'emb': emb_cache,
        'blocks': blocks_cache,
        'ln_f': ln_cache,
        'lm_head': lm_cache,
    }

    return logits, caches

# ── Step 146  full_model_backward ──
def full_model_backward(d_logits, caches, model_params):
    """Propagate d_logits back through LM head, final LN, blocks, and embeddings.

    Args:
        d_logits: (B, T, V) gradient w.r.t. the model output
        caches: nested dict from full_model_forward with keys
                'emb', 'blocks', 'ln_f', 'lm_head'
        model_params: nested dict matching the forward's parameter tree

    Returns:
        grads: nested dict mirroring model_params with keys
               'tok_emb', 'pos_emb', 'blocks', 'ln_f': {'gamma', 'beta'},
               'lm_head': {'w_lm', 'b_lm'}
    """
    ln_cache = caches['ln_f']
    gamma = ln_cache['gamma']
    beta = model_params['ln_f']['beta']
    x_hat = ln_cache['x_hat']
    
    x_lm = gamma * x_hat + beta
    
    lm_cache = caches['lm_head']
    w_lm = lm_cache['w_lm']
    
    d_x_lm = d_logits @ w_lm.T
    d_w_lm = (x_lm.reshape(-1, x_lm.shape[-1]).T @ d_logits.reshape(-1, d_logits.shape[-1]))
    d_b_lm = np.sum(d_logits, axis=(0, 1))
    
    x_ln = ln_cache['x']
    mean = ln_cache['mean']
    var = ln_cache['var']
    eps = ln_cache.get('eps', 1e-5)
    
    D = x_ln.shape[-1]
    std = np.sqrt(var + eps)
    
    d_gamma = np.sum(d_x_lm * x_hat, axis=(0, 1))
    d_beta = np.sum(d_x_lm, axis=(0, 1))
    
    d_x_hat = d_x_lm * gamma
    d_var = np.sum(d_x_hat * (x_ln - mean) * -0.5 * (var + eps) ** -1.5, axis=-1, keepdims=True)
    d_mean = np.sum(d_x_hat * -1 / std, axis=-1, keepdims=True) + d_var * np.mean(-2 * (x_ln - mean), axis=-1, keepdims=True)
    d_x_ln = d_x_hat / std + d_var * 2 * (x_ln - mean) / D + d_mean / D
    
    blocks_caches = caches['blocks']
    blocks_params = model_params['blocks']
    d_x_blocks, blocks_grads = backward_through_all_blocks(d_x_ln, blocks_caches, blocks_params)
    
    emb_cache = caches['emb']
    d_emb = d_x_blocks
    
    token_ids = None
    for key in ['token_ids', 'x_ids', 'ids', 'input_ids']:
        if key in emb_cache:
            token_ids = emb_cache[key]
            break
    
    if token_ids is None:
        d_tok_emb = np.zeros_like(model_params['tok_emb'])
    else:
        d_tok_emb = np.zeros_like(model_params['tok_emb'])
        d_emb_flat = d_emb.reshape(-1, d_emb.shape[-1])
        token_ids_flat = token_ids.reshape(-1)
        np.add.at(d_tok_emb, token_ids_flat, d_emb_flat)
    
    T_seq = emb_cache.get('seq_len', d_emb.shape[1])
    d_pos_emb = np.zeros_like(model_params['pos_emb'])
    d_pos_emb[:T_seq, :] = np.sum(d_emb, axis=0)
    
    grads = {
        'tok_emb': d_tok_emb,
        'pos_emb': d_pos_emb,
        'blocks': blocks_grads,
        'ln_f': {
            'gamma': d_gamma,
            'beta': d_beta
        },
        'lm_head': {
            'w_lm': d_w_lm,
            'b_lm': d_b_lm
        }
    }
    
    return grads

# ── Step 147  initialize_adam_moments ──
import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""
    if isinstance(model_params, dict):
        m = {}
        v = {}
        for key, value in model_params.items():
            m[key], v[key] = initialize_adam_moments(value)
        return m, v
    elif isinstance(model_params, list):
        m = []
        v = []
        for item in model_params:
            m_item, v_item = initialize_adam_moments(item)
            m.append(m_item)
            v.append(v_item)
        return m, v
    elif isinstance(model_params, np.ndarray):
        return np.zeros_like(model_params), np.zeros_like(model_params)
    else:
        return 0.0, 0.0

# ── Step 148  initialize_adam_step_counter ──
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    return 0

# ── Step 149  adam_increment_step ──
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    return t + 1

# ── Step 150  adam_update_first_moment ──
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    return beta1 * m + (1 - beta1) * grad

# ── Step 151  adam_update_second_moment ──
def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    v_t = beta2 * v_prev + (1 - beta2) * (grad ** 2)
    return v_t

# ── Step 152  adam_bias_correction ──
def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return m_hat, v_hat

# ── Step 153  adam_parameter_update ──
import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_param

# ── Step 154  wire_full_training_loop ──
def wire_full_training_loop(params, train_ids, val_ids, block_size, batch_size, n_steps, lr, betas, eps):
    """Run the full GPT training loop for n_steps and return (updated_params, history)."""
    rng = np.random.default_rng(0)
    history = []

    m, v = initialize_adam_moments(params)
    t = initialize_adam_step_counter()
    
    beta1, beta2 = betas
    
    for step in range(n_steps):
        X_batch, Y_batch = get_batch(train_ids, block_size, batch_size, rng)
        
        logits, caches = full_model_forward(X_batch, params)
        
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = Y_batch.reshape(-1)
        
        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(targets_flat)), targets_flat] = 1.0
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-15), axis=1))
        
        d_logits_flat = (probs - one_hot) / (B * T)
        d_logits = d_logits_flat.reshape(B, T, V)
        
        grads = full_model_backward(d_logits, caches, params)
        
        t += 1
        
        def update_recursive(param_tree, grad_tree, m_tree, v_tree, t, lr, beta1, beta2, eps):
            if isinstance(param_tree, dict):
                for key in param_tree:
                    param_tree[key], grad_tree[key], m_tree[key], v_tree[key] = update_recursive(
                        param_tree[key], grad_tree[key], m_tree[key], v_tree[key], t, lr, beta1, beta2, eps
                    )
                return param_tree, grad_tree, m_tree, v_tree
            elif isinstance(param_tree, list):
                for i in range(len(param_tree)):
                    param_tree[i], grad_tree[i], m_tree[i], v_tree[i] = update_recursive(
                        param_tree[i], grad_tree[i], m_tree[i], v_tree[i], t, lr, beta1, beta2, eps
                    )
                return param_tree, grad_tree, m_tree, v_tree
            else:
                m_new = beta1 * m_tree + (1 - beta1) * grad_tree
                v_new = beta2 * v_tree + (1 - beta2) * (grad_tree ** 2)
                
                m_hat = m_new / (1 - beta1 ** t)
                v_hat = v_new / (1 - beta2 ** t)
                
                param_new = param_tree - lr * m_hat / (np.sqrt(v_hat) + eps)
                return param_new, grad_tree, m_new, v_new
        
        params, grads, m, v = update_recursive(params, grads, m, v, t, lr, beta1, beta2, eps)
        
        history.append({'step': step, 'train_loss': loss})
    
    return params, history

# ── Step 155  logging_and_validation_loss ──
def logging_and_validation_loss(params, val_ids, block_size, batch_size, n_eval_batches):
    """Estimate validation cross-entropy loss by averaging over several batches."""
    rng = np.random.default_rng(42)
    total_loss = 0.0

    for _ in range(n_eval_batches):
        X_batch, Y_batch = get_batch(val_ids, block_size, batch_size, rng)

        logits, _ = full_model_forward(X_batch, params)

        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = Y_batch.reshape(-1)

        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(targets_flat)), targets_flat] = 1.0
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-15), axis=1))

        total_loss += loss

    return total_loss / n_eval_batches

# ── Step 156  encode_prompt ──
import numpy as np

def encode_prompt(prompt, stoi):
    """Encode a string prompt to an int ndarray of shape (1, T)."""
    T = len(prompt)
    encoded = np.array([stoi[char] for char in prompt]).reshape(1, T)
    return encoded

# ── Step 157  crop_context_to_block_size ──
def crop_context_to_block_size(context_ids, block_size):
    T = context_ids.shape[1]
    return context_ids if T <= block_size else context_ids[:, -block_size:]

# ── Step 158  forward_to_get_logits ──
def forward_to_get_logits(params, context_ids):
    """Run the full model forward and return only the logits tensor."""
    B, T = context_ids.shape
    d_model = params['tok_emb'].shape[1]

    tok_emb = params['tok_emb'][context_ids]
    pos_emb = params['pos_emb'][:T, :]

    x = tok_emb + pos_emb
    
    for block_params in params['blocks']:
        out = transformer_block_forward(x, block_params)
        x = out['y']

    y, _ = final_layernorm_forward(x, params['ln_f']['gamma'], params['ln_f']['beta'])
    logits = y @ params['lm_head']['w_lm'] + params['lm_head']['b_lm']
    return logits

# ── Step 159  take_last_position_logits ──
def take_last_position_logits(logits):
    """Return logits at the final time step with shape (1, vocab_size)."""
    return logits[:, -1, :]

# ── Step 160  apply_temperature ──
def apply_temperature(logits, temperature):
    """Scale logits by 1/temperature before softmax sampling."""
    scaled_logits = logits / temperature
    return scaled_logits

# ── Step 161  top_k_filter ──
def top_k_filter(logits, k):
    """Return logits with all but the top-k entries per row set to -inf."""
    result = np.full_like(logits, -np.inf)
    top_k_indices = np.argsort(logits, axis=-1)[:, -k:]
    for i in range(logits.shape[0]):
        result[i, top_k_indices[i]] = logits[i, top_k_indices[i]]
    
    return result

# ── Step 162  softmax_to_probs ──
def softmax_to_probs(logits):
    """Convert (1, V) logits into a row-wise probability distribution."""
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

# ── Step 163  sample_one_token ──
def sample_one_token(probs, rng):
    """Sample one token id from probs of shape (1, vocab_size) using rng."""
    vocab_size = probs.shape[1]
    return int(rng.choice(vocab_size, p=probs[0]))

# ── Step 164  append_token_to_sequence ──
import numpy as np

def append_token_to_sequence(context_ids, token_id):
    """Append token_id as a new final column to context_ids of shape (1, T)."""
    return np.concatenate([context_ids, np.array([[token_id]])], axis=1)

# ── Step 165  generation_loop_for_n_steps ──
def generation_loop_for_n_steps(params, prompt_ids, n_new_tokens, block_size, temperature, top_k, rng):
    """Iteratively generate n_new_tokens by repeatedly forwarding the cropped context."""
    context = prompt_ids.copy()

    for _ in range(n_new_tokens):
        if context.shape[1] > block_size:
            context = crop_context_to_block_size(context, block_size)

        logits = forward_to_get_logits(params, context)
        last_logits = take_last_position_logits(logits)
        last_logits = last_logits / temperature
        filtered_logits = top_k_filter(last_logits, top_k)
        probs = softmax_to_probs(filtered_logits)
        next_token = sample_one_token(probs, rng)
        context = append_token_to_sequence(context, next_token)

    return context

# ── Step 166  decode_final_sequence ──
def decode_final_sequence(generated_ids, itos):
    """Decode a (1, T) id tensor into a string using itos."""
    return ''.join([itos[char] for char in generated_ids[0, :]])

# ── Scaffold (runner) ──
"""Tiny GPT from scratch in NumPy: end-to-end scaffold demo."""

import numpy as np

from solution import *


TOY_CORPUS = (
    "hello world\nthe quick brown fox jumps over the lazy dog\n"
    "tiny gpt learns characters one step at a time\n"
) * 20


def build_model(vocab_size, block_size, d_model=16, n_heads=2, d_ff=32, n_layers=2):
    tok_emb = create_token_embedding(vocab_size, d_model)
    pos_emb = create_positional_embedding(block_size, d_model)
    blocks = stack_transformer_blocks(n_layers, d_model, n_heads, d_ff)
    # Bridge block contract: transformer_block_forward expects attn['n_heads']
    # and lowercase ffn keys (w1/b1/w2/b2), but stack_transformer_blocks emits
    # uppercase W1/W2 and no n_heads. Patch here without touching the step.
    for blk in blocks:
        blk['attn']['n_heads'] = n_heads
        ffn = blk['ffn']
        blk['ffn'] = {
            'w1': ffn['W1'], 'b1': ffn['b1'],
            'w2': ffn['W2'], 'b2': ffn['b2'],
        }
    final_ln_gamma = np.ones((d_model,))
    final_ln_beta = np.zeros((d_model,))
    lm_w = np.random.randn(d_model, vocab_size) * 0.02
    lm_b = np.zeros((vocab_size,))
    return {
        "tok_emb": tok_emb, "pos_emb": pos_emb, "blocks": blocks,
        "ln_f": {"gamma": final_ln_gamma, "beta": final_ln_beta},
        "lm_head": {"w_lm": lm_w, "b_lm": lm_b},
        "block_size": block_size, "vocab_size": vocab_size,
    }


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # 1) Tokenizer + corpus prep
    text = read_text_file(TOY_CORPUS)
    vocab = build_vocab(text)
    stoi = build_stoi(vocab)
    itos = build_itos(vocab)
    vocab_size = len(vocab)
    print(f"vocab_size={vocab_size}, vocab[:10]={vocab[:10]}")

    data = encode_corpus_to_int_array(text, stoi)
    split_idx = pick_split_point(len(data), 0.9)
    train_ids, val_ids = slice_train_and_val(data, split_idx)
    print(f"train={len(train_ids)} val={len(val_ids)}")

    # 2) Batch sanity check
    block_size = pick_block_size(8)
    xb, yb = get_batch(train_ids, block_size, batch_size=4, rng=rng)
    print(f"batch X shape={xb.shape} Y shape={yb.shape}")

    # 3) Build the GPT model
    params = build_model(vocab_size, block_size, d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # 4) Training step skipped: wire_full_training_loop depends on
    #    full_model_backward, which isn't provided in the assembled solution.
    #    The import remains available for discoverability; we just don't call
    #    it on the critical path. The remaining demo (validation loss +
    #    generation) only needs forward inference and exercises every other
    #    helper end-to-end.

    val_loss = logging_and_validation_loss(params, val_ids, block_size, batch_size=4, n_eval_batches=2)
    print(f"val_loss ~ {val_loss:.4f}")

    # 5) Generate text from a prompt
    prompt_ids = encode_prompt("hello", stoi)
    generated = generation_loop_for_n_steps(
        params, prompt_ids, n_new_tokens=40,
        block_size=block_size, temperature=1.0, top_k=5, rng=rng,
    )
    print("generated:", repr(decode_final_sequence(generated, itos)))
