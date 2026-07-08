"""
DiLoCo: Distributed Low-Communication Training of Language Models — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  init_model_params ──
def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# ── Step 002  relu ──
import numpy as np

def relu(x):
    return np.maximum(x, 0)

# ── Step 003  model_forward ──
import numpy as np

def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    z1 = x @ W1 + b1
    h1 = np.maximum(0, z1)
    logits = h1 @ W2 + b2
    cache = {'x': x, 'z1': z1, 'h1': h1, 'logits': logits}
    return logits, cache

# ── Step 004  softmax ──
import numpy as np

def softmax(logits):
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(logits - max_logits)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

# ── Step 005  cross_entropy_loss ──
def cross_entropy_loss(logits, labels, eps=1e-12):
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_softmax = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    
    N = logits.shape[0]
    loss = 0.0
    for i in range(N):
        loss += -log_softmax[i, labels[i]]
        
    return loss / N

# ── Step 006  model_backward ──
def model_backward(params, cache, labels):
    x, z1, h1, logits = cache['x'], cache['z1'], cache['h1'], cache['logits']
    N = x.shape[0]
    C = logits.shape[0]

    probs = softmax(logits)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    dlogits = (probs - one_hot) / N

    db2 = np.sum(dlogits, axis=0)
    dW2 = h1.T @ dlogits
    dh1 = dlogits @ params['W2'].T

    dz1 = dh1 * (z1 > 0)

    db1 = np.sum(dz1, axis=0)
    dW1 = x.T @ dz1

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

# ── Step 007  init_adamw_state ──
def init_adamw_state(params):
    m, v = {}, {}
    for key, arr in params.items():
        m[key] = np.zeros_like(arr)
        v[key] = np.zeros_like(arr)

    return {'m': m, 'v': v, 't': 0}

# ── Step 008  update_adam_moments ──
def update_adam_moments(state, grads, beta1, beta2):
    state['t'] += 1

    for key in grads:
        state['m'][key] = beta1 * state['m'][key] + (1 - beta1) * grads[key]
        state['v'][key] = beta2 * state['v'][key] + (1 - beta2) * (grads[key] ** 2)

    return state

# ── Step 009  bias_correct_moments ──
def bias_correct_moments(state, beta1, beta2):
    t = state['t']
    m_hat = {}
    v_hat = {}
    for key in state['m']:
        m_hat[key] = state['m'][key] / (1 - beta1 ** t)
        v_hat[key] = state['v'][key] / (1 - beta2 ** t)

    return m_hat, v_hat

# ── Step 010  adam_param_step ──
def adam_param_step(params, m_hat, v_hat, lr, eps):
    new_params = {}
    for key in params:
        new_params[key] = params[key] - lr * m_hat[key] / (np.sqrt(v_hat[key]) + eps)

    return new_params

# ── Step 011  decoupled_weight_decay ──
import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    new_params = {}
    decay_factor = 1.0 - lr * weight_decay
    for key in params:
        new_params[key] = params[key] * decay_factor
    
    return new_params

# ── Step 012  clone_params ──
def clone_params(params):
    return {key: value.copy() for key, value in params.items()}

# ── Step 013  scale_params ──
def scale_params(params, scalar):
    return {key: value * scalar for key, value in params.items()}

# ── Step 014  subtract_params ──
def subtract_params(params_a, params_b):
    return {key: params_a[key] - params_b[key] for key in params_a}

# ── Step 015  average_params ──
def average_params(params_list):
    avg = {}
    for key in params_list[0]:
        avg[key] = np.mean([p[key] for p in params_list], axis=0)

    return avg

# ── Step 016  iid_shard_dataset ──
def iid_shard_dataset(x, y, num_workers, seed=0):
    np.random.seed(seed)
    N = x.shape[0]
    indices = np.random.permutation(N)

    shards = []
    base_size = N // num_workers
    remainder = N % num_workers

    start = 0
    for i in range(num_workers):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        shard_indices = indices[start:end]
        shards.append((x[shard_indices], y[shard_indices]))
        start = end

    return shards

# ── Step 017  noniid_shard_dataset ──
def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    class_to_worker = {}
    for c in range(num_classes):
        class_to_worker[c] = c % num_workers

    worker_indices = [[] for _ in range(num_workers)]
    for idx, label in enumerate(y):
        worker = class_to_worker[label]
        worker_indices[worker].append(idx)

    rng = np.random.default_rng(seed)
    shards = []
    for worker in range(num_workers):
        indices = np.array(worker_indices[worker])
        rng.shuffle(indices)
        shards.append((x[indices], y[indices]))

    return shards

# ── Step 018  sample_worker_batch ──
def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    n = x_shard.shape[0]
    replace = batch_size > n
    indices = rng.choice(n, size=batch_size, replace=replace)
    return x_shard[indices], y_shard[indices]

# ── Step 019  local_train_step ──
def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    logits, cache = model_forward(params, x_batch)
    loss = cross_entropy_loss(logits, y_batch)
    grads = model_backward(params, cache, y_batch)
    adam_state = update_adam_moments(adam_state, grads, beta1, beta2)
    m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
    new_params = adam_param_step(params, m_hat, v_hat, lr, eps)
    new_params = decoupled_weight_decay(new_params, lr, weight_decay)
    return new_params, adam_state, loss

# ── Step 020  inner_train_worker ──
def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    worker_params = clone_params(params)
    adam_state = init_adamw_state(worker_params)
    rng = np.random.default_rng(seed)
    total_loss = 0.0
    
    for _ in range(num_inner_steps):
        x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
        worker_params, adam_state, loss = local_train_step(
            worker_params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay
        )
        total_loss += loss
    
    if num_inner_steps > 0:
        mean_loss = total_loss / num_inner_steps
    else:
        mean_loss = 0.0
    
    return worker_params, mean_loss

# ── Step 021  init_outer_optimizer ──
def init_outer_optimizer(params):
    momentum = {}
    for key in params:
        momentum[key] = np.zeros_like(params[key])

    return {'momentum': momentum}

# ── Step 022  update_outer_momentum ──
import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    """Update Nesterov momentum buffer: m <- momentum_coef * m + outer_grad."""
    for key in outer_state['momentum']:
        outer_state['momentum'][key] = momentum_coef * outer_state['momentum'][key] + outer_grad[key]
    
    return outer_state

# ── Step 023  nesterov_param_update ──
def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    if 'momentum' in outer_state:
        momentum = outer_state['momentum']
    else:
        momentum = outer_state
    
    new_params = {}
    for key in params:
        new_params[key] = params[key] - outer_lr * (momentum_coef * momentum[key] + outer_grad[key])
        
    return new_params

# ── Step 024  compute_outer_gradient ──
def compute_outer_gradient(global_params, worker_params_list):
    avg_worker = average_params(worker_params_list)
    return subtract_params(global_params, avg_worker)

# ── Step 025  run_diloco_round ──
def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']
    
    worker_params_list = []
    worker_losses = []
    
    for worker_idx, (x_shard, y_shard) in enumerate(worker_shards):
        worker_seed = seed + worker_idx
        worker_params, mean_loss = inner_train_worker(
            global_params, x_shard, y_shard, num_inner_steps, batch_size,
            lr, beta1, beta2, eps, weight_decay, worker_seed
        )
        worker_params_list.append(worker_params)
        worker_losses.append(mean_loss)
    
    outer_grad = compute_outer_gradient(global_params, worker_params_list)
    outer_state = update_outer_momentum(outer_state, outer_grad, momentum_coef)
    new_global_params = nesterov_param_update(global_params, outer_state, outer_grad, outer_lr, momentum_coef)
    return new_global_params, outer_state, worker_losses

# ── Step 026  train_diloco ──
def train_diloco(init_params, worker_shards, num_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed=0):
    global_params = clone_params(init_params)
    outer_state = init_outer_optimizer(global_params)
    
    round_losses = []
    
    for round_idx in range(num_rounds):
        round_seed = seed + round_idx
        global_params, outer_state, worker_losses = run_diloco_round(
            global_params, outer_state, worker_shards, num_inner_steps, batch_size,
            inner_hparams, outer_lr, momentum_coef, round_seed
        )
        mean_round_loss = np.mean(worker_losses)
        round_losses.append(mean_round_loss)
    
    return global_params, {'round_losses': round_losses}

# ── Step 027  train_synchronous_baseline ──
def train_synchronous_baseline(init_params, worker_shards, num_steps, batch_size, inner_hparams, seed=0):
    params = clone_params(init_params)
    
    adam_state = init_adamw_state(params)
    
    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']
    
    rng = np.random.default_rng(seed)
    
    step_losses = []
    num_workers = len(worker_shards)
    
    for step in range(num_steps):
        grad_list = []
        loss_list = []
        
        for worker_idx, (x_shard, y_shard) in enumerate(worker_shards):
            x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
            logits, cache = model_forward(params, x_batch)
            loss = cross_entropy_loss(logits, y_batch)
            loss_list.append(loss)
            grads = model_backward(params, cache, y_batch)
            grad_list.append(grads)
        
        avg_grads = average_params(grad_list)
        adam_state = update_adam_moments(adam_state, avg_grads, beta1, beta2)
        m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
        params = adam_param_step(params, m_hat, v_hat, lr, eps)
        params = decoupled_weight_decay(params, lr, weight_decay)
        step_losses.append(np.mean(loss_list))
    
    return params, {'step_losses': step_losses}

# ── Step 028  evaluate_loss ──
def evaluate_loss(params, x, y):
    logits, _ = model_forward(params, x)
    return float(cross_entropy_loss(logits, y))

# ── Step 029  classification_accuracy ──
def classification_accuracy(params, x, y):
    logits, _ = model_forward(params, x)
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == y))

# ── Step 030  communication_savings ──
def communication_savings(num_rounds, num_inner_steps, num_workers, param_count):
    per_event = 2 * num_workers * param_count
    diloco_scalars = num_rounds * per_event
    sync_scalars = num_rounds * num_inner_steps * per_event
    return {
        'diloco_scalars': diloco_scalars,
        'sync_scalars': sync_scalars,
        'ratio': diloco_scalars / sync_scalars,
        'savings_factor': sync_scalars / diloco_scalars,
    }

# ── Scaffold (runner) ──
"""End-to-end demo of DiLoCo: distributed low-communication training vs. sync baseline."""

import numpy as np


def main():
    np.random.seed(0)

    # Synthetic classification dataset
    num_samples, input_dim, hidden_dim, num_classes = 400, 8, 16, 3
    X = np.random.randn(num_samples, input_dim).astype(np.float64)
    true_W = np.random.randn(input_dim, num_classes)
    y = np.argmax(X @ true_W + 0.1 * np.random.randn(num_samples, num_classes), axis=1)

    # Held-out eval split
    n_train = 320
    x_train, y_train = X[:n_train], y[:n_train]
    x_eval, y_eval = X[n_train:], y[n_train:]

    # Initial (shared) model params
    init_params = init_model_params(input_dim, hidden_dim, num_classes, seed=0)
    print("Initialized 2-layer MLP with param keys:", sorted(init_params.keys()))

    # Sanity-check forward + loss
    logits, _ = model_forward(init_params, x_train[:4])
    print("Initial logits shape:", logits.shape)
    print("Initial train loss:", round(evaluate_loss(init_params, x_train, y_train), 4))
    print("Initial train acc :", round(classification_accuracy(init_params, x_train, y_train), 4))

    # Shard the data across workers (IID)
    num_workers = 4
    worker_shards = iid_shard_dataset(x_train, y_train, num_workers, seed=1)
    print(f"\nIID shards: {[len(sy) for _, sy in worker_shards]}")

    # DiLoCo hyperparameters
    inner_hparams = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4)
    num_rounds = 8
    num_inner_steps = 20
    batch_size = 16
    outer_lr = 0.7
    momentum_coef = 0.9

    # --- Run DiLoCo ---
    diloco_params, diloco_log = train_diloco(
        clone_params(init_params),
        worker_shards,
        num_rounds=num_rounds,
        num_inner_steps=num_inner_steps,
        batch_size=batch_size,
        inner_hparams=inner_hparams,
        outer_lr=outer_lr,
        momentum_coef=momentum_coef,
        seed=42,
    )
    print("\n=== DiLoCo training log (last few rounds) ===")
    if isinstance(diloco_log, dict):
        log_entries = [{k: diloco_log[k][i] for k in diloco_log} for i in range(len(next(iter(diloco_log.values()))))]
    else:
        log_entries = list(diloco_log)
    for entry in log_entries[-3:]:
        print(entry)
    print("DiLoCo eval loss:", round(evaluate_loss(diloco_params, x_eval, y_eval), 4))
    print("DiLoCo eval acc :", round(classification_accuracy(diloco_params, x_eval, y_eval), 4))

    # --- Synchronous baseline: communicates every step ---
    total_sync_steps = num_rounds * num_inner_steps
    sync_result = train_synchronous_baseline(
        clone_params(init_params),
        worker_shards,
        num_steps=total_sync_steps,
        batch_size=batch_size,
        inner_hparams=inner_hparams,
        seed=42,
    )
    if isinstance(sync_result, tuple):
        sync_params = sync_result[0]
    else:
        sync_params = sync_result
    print("\n=== Synchronous baseline ===")
    print("Sync   eval loss:", round(evaluate_loss(sync_params, x_eval, y_eval), 4))
    print("Sync   eval acc :", round(classification_accuracy(sync_params, x_eval, y_eval), 4))

    # --- Communication accounting ---
    param_count = sum(int(np.asarray(v).size) for v in init_params.values())
    savings = communication_savings(num_rounds, num_inner_steps, num_workers, param_count)
    print("\n=== Communication accounting ===")
    print(f"Total params per worker : {param_count}")
    print(f"Communication summary   : {savings}")


if __name__ == "__main__":
    main()
