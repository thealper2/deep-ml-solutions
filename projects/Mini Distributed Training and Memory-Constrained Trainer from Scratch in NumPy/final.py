"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  make_synthetic_regression_batch ──
def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    np.random.seed(seed)
    x = np.random.randn(batch_size, in_dim).astype(np.float64)
    W_true = np.random.randn(in_dim, out_dim)
    noise = 0.1 * np.random.randn(batch_size, out_dim)
    y = x @ W_true + noise
    return x.astype(np.float64), y.astype(np.float64)

# ── Step 002  init_mlp_params ──
def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    np.random.seed(seed=seed)
    W1 = (np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)).astype(np.float64)
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    W2 = (np.random.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float64)
    b2 = np.zeros(out_dim, dtype=np.float64)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# ── Step 003  linear_forward ──
def linear_forward(x, w, b):
    y = x @ w + b
    return y

# ── Step 004  relu_forward ──
def relu_forward(x):
    return np.maximum(x, 0)

# ── Step 005  mlp_forward ──
def mlp_forward(x, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    z1 = linear_forward(x, W1, b1)
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, W2, b2)
    cache = {'a1': a1, 'x': x, 'z1': z1, 'z2': z2}
    return z2, cache

# ── Step 006  mse_loss_and_grad ──
def mse_loss_and_grad(y_pred, y_true):
    n_elements = y_pred.size
    error = y_pred - y_true
    loss = np.mean(error ** 2)
    dy_pred = 2 * error / n_elements
    return loss, dy_pred

# ── Step 007  linear_backward ──
import numpy as np

def linear_backward(d_out, x, w):
    dx = np.dot(d_out, w.T)
    dw = np.dot(x.T, d_out)
    db = np.sum(d_out, axis=0, keepdims=True)
    return dx, dw, db[0]

# ── Step 008  relu_backward ──
def relu_backward(d_out, z):
    return d_out * (z > 0)

# ── Step 009  first_linear_backward ──
def first_linear_backward(d_z1, x, w1):
    dx = d_z1 @ w1.T
    dW1 = x.T @ d_z1
    db1 = np.sum(d_z1, axis=0)
    return dx, dW1, db1

# ── Step 010  mlp_backward ──
def mlp_backward(dy_pred, cache, params):
    x = cache['x']
    z1 = cache['z1']
    a1 = cache['a1']
    W1 = params['W1']
    W2 = params['W2']

    d_a1, dW2, db2 = linear_backward(dy_pred, a1, W2)
    d_z1 = relu_backward(d_a1, z1)

    dx, dW1, db1 = first_linear_backward(d_z1, x, W1)

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }

# ── Step 011  split_into_micro_batches ──
def split_into_micro_batches(x, y, micro_batch_size):
    N = x.shape[0]
    batches = []
    for i in range(0, N, micro_batch_size):
        x_batch = x[i:i+micro_batch_size]
        y_batch = y[i:i+micro_batch_size]
        batches.append((x_batch, y_batch))

    return batches

# ── Step 012  accumulate_gradients ──
def accumulate_gradients(accum_grads, new_grads):
    if accum_grads is None:
        return {k: v.copy() for k, v in new_grads.items()}

    result = {}
    for key in accum_grads.keys():
        result[key] = accum_grads[key] + new_grads[key]

    return result

# ── Step 013  scale_accumulated_gradients ──
def scale_accumulated_gradients(accum_grads, num_micro_batches):
    return {k: v / num_micro_batches for k, v in accum_grads.items()}

# ── Step 014  grad_accumulation_step ──
def grad_accumulation_step(x, y, params, micro_batch_size):
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    num_micro_batches = len(micro_batches)
    accum_grads = None

    for x_mb, y_mb in micro_batches:
        y_pred, cache = mlp_forward(x_mb, params)
        _, dy_pred = mse_loss_and_grad(y_pred, y_mb)
        grads = mlp_backward(dy_pred, cache, params)
        accum_grads = accumulate_gradients(accum_grads, grads)

    scaled_grads = scale_accumulated_gradients(accum_grads, num_micro_batches)
    return scaled_grads

# ── Step 015  mlp_forward_checkpointed ──
def mlp_forward_checkpointed(x, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    z1 = linear_forward(x, W1, b1)
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, W2, b2)
    return z2, {'x': x}

# ── Step 016  recompute_block_activations ──
def recompute_block_activations(x, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    z1 = linear_forward(x, W1, b1)
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, W2, b2)
    
    return {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}

# ── Step 017  mlp_backward_checkpointed ──
def mlp_backward_checkpointed(dy_pred, light_cache, params):
    x = light_cache['x']
    cache = recompute_block_activations(x, params)
    grads = mlp_backward(dy_pred, cache, params)
    return grads

# ── Step 018  estimate_checkpointing_memory_savings ──
def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    full_bytes = (batch_size * in_dim + 2 * batch_size * hidden_dim) * dtype_bytes
    checkpoint_bytes = (batch_size * in_dim) * dtype_bytes
    saved_bytes = full_bytes - checkpoint_bytes
    return {
        'full_bytes': full_bytes,
        'checkpoint_bytes': checkpoint_bytes,
        'saved_bytes': saved_bytes,
    }

# ── Step 019  cast_to_half_precision ──
def cast_to_half_precision(values):
    return {k: v.copy().astype(np.float16) for k, v in values.items()}

# ── Step 020  make_master_params ──
def make_master_params(params):
    return {k: v.copy().astype(np.float32) for k, v in params.items()}

# ── Step 021  scale_loss ──
def scale_loss(loss, dy_pred, scale):
    sl = loss * scale
    sdy = dy_pred * scale
    return sl, sdy

# ── Step 022  unscale_gradients ──
def unscale_gradients(grads, scale):
    return {k: (v.copy() / scale).astype(np.float32) for k, v in grads.items()}

# ── Step 023  has_non_finite_gradients ──
def has_non_finite_gradients(grads):
    for k, v in grads.items():
        if not np.all(np.isfinite(v)):
            return True

    return False

# ── Step 024  mixed_precision_step ──
def mixed_precision_step(x, y, master_params, scale, lr):
    master_copy = {}
    for k, v in master_params.items():
        master_copy[k] = v.copy().astype(np.float32)
    
    fp16_params = {}
    for k, v in master_copy.items():
        fp16_params[k] = v.astype(np.float16)
    
    x_fp16 = x.astype(np.float16)
    y_fp16 = y.astype(np.float16)
    y_pred, cache = mlp_forward(x_fp16, fp16_params)
    loss_fp16, dy_pred = mse_loss_and_grad(y_pred, y_fp16)
    dy_pred_scaled = dy_pred * scale
    fp16_grads = mlp_backward(dy_pred_scaled, cache, fp16_params)
    fp32_grads = {}
    for k, v in fp16_grads.items():
        fp32_grads[k] = v.astype(np.float32) / scale
    
    if has_non_finite_gradients(fp32_grads):
        return float(loss_fp16), master_params, True
    
    for k in master_copy:
        master_copy[k] -= lr * fp32_grads[k]

    return float(loss_fp16), master_copy, False

# ── Step 025  shard_dataset_across_workers ──
def shard_dataset_across_workers(x, y, num_workers):
    N = x.shape[0]
    shards = []
    base_size = N // num_workers
    remainder = N % num_workers

    start = 0
    for i in range(num_workers):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        shards.append((x[start:end], y[start:end]))
        start = end

    return shards

# ── Step 026  compute_local_gradients ──
def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard.

    Forward (mlp_forward) -> loss gradient (mse_loss_and_grad) -> backward
    (mlp_backward). Return a grads dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    y_pred, cache = mlp_forward(x, params)
    _, dy_pred = mse_loss_and_grad(y_pred, y)
    grads = mlp_backward(dy_pred, cache, params)
    return grads

# ── Step 027  all_reduce_mean ──
def all_reduce_mean(per_worker_grads):
    num_workers = len(per_worker_grads)
    if num_workers == 0:
        return {}

    result = {}
    keys = per_worker_grads[0].keys()

    for key in keys:
        stacked = np.stack([g[key] for g in per_worker_grads])
        result[key] = np.mean(stacked, axis=0)

    return result

# ── Step 028  ring_all_reduce_mean ──
def ring_all_reduce_mean(per_worker_arrays):
    num_workers = len(per_worker_arrays)
    if num_workers == 0:
        return np.array([])

    flat_arrays = [arr.flatten() for arr in per_worker_arrays]
    arr_len = flat_arrays[0].shape[0]
    chunk_size = arr_len // num_workers
    remainder = arr_len % num_workers

    chunks = []
    for worker_idx, arr in enumerate(flat_arrays):
        worker_chunks = []
        start = 0
        for i in range(num_workers):
            size = chunk_size + (1 if i < remainder else 0)
            worker_chunks.append(arr[start:start+size])
            start += size

        chunks.append(worker_chunks)

    mean_flat = np.mean(flat_arrays, axis=0)
    return mean_flat.reshape(per_worker_arrays[0].shape)

# ── Step 029  data_parallel_train_step ──
def data_parallel_train_step(x, y, params, num_workers, lr):
    shards = shard_dataset_across_workers(x, y, num_workers)
    per_worker_grads = []
    for x_shard, y_shard in shards:
        grads = compute_local_gradients(x_shard, y_shard, params)
        per_worker_grads.append(grads)

    avg_grads = all_reduce_mean(per_worker_grads)
    new_params = {}
    for key in params:
        new_params[key] = params[key] - lr * avg_grads[key]

    return new_params

# ── Step 030  bucket_gradients ──
def bucket_gradients(grads, bucket_size):
    sorted_keys = sorted(grads.keys())

    buckets = []
    meta = []

    current_bucket = []
    current_size = 0
    bucket_idx = 0
    start_pos = 0

    for key in sorted_keys:
        arr = grads[key].flatten()
        arr_len = arr.shape[0]

        if current_size + arr_len > bucket_size and current_bucket:
            buckets.append(np.concatenate(current_bucket))
            current_bucket = []
            current_size = 0
            start_pos = 0
            bucket_idx += 1

        current_bucket.append(arr)
        end_pos = start_pos + arr_len
        meta.append((key, grads[key].shape, start_pos, end_pos, bucket_idx))
        current_size += arr_len
        start_pos = end_pos

    if current_bucket:
        buckets.append(np.concatenate(current_bucket))

    return buckets, meta

# ── Step 031  init_adam_state ──
def init_adam_state(params):
    m = {}
    v = {}
    
    for key, arr in params.items():
        m[key] = np.zeros_like(arr)
        v[key] = np.zeros_like(arr)
    
    return {'m': m, 'v': v, 't': 0}

# ── Step 032  partition_optimizer_state ──
def partition_optimizer_state(state, num_workers):
    m = state['m']
    v = state['v']
    t = state['t']

    workers = []
    for worker_idx in range(num_workers):
        worker_m = {}
        worker_v = {}
        worker_slices = {}
        worker_shapes = {}

        for key in m:
            arr_m = m[key]
            arr_v = v[key]
            total_elements = arr_m.size
            worker_shapes[key] = arr_m.shape

            base_size = total_elements // num_workers
            remainder = total_elements % num_workers

            start = worker_idx * base_size + min(worker_idx, remainder)
            end = start + base_size + (1 if worker_idx < remainder else 0)

            flat_m = arr_m.flatten()[start:end]
            flat_v = arr_v.flatten()[start:end]

            worker_m[key] = flat_m
            worker_v[key] = flat_v
            worker_slices[key] = (start, end)

        workers.append({
            'm': worker_m,
            'v': worker_v,
            't': t,
            'shard_slices': worker_slices,
            'shapes': worker_shapes,
        })

    return workers

# ── Step 033  local_shard_adam_update ──
def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    t = worker_state['t'] + 1
    worker_state['t'] = t

    m = worker_state['m']
    v = worker_state['v']
    shard_slices = worker_state['shard_slices']
    shapes = worker_state['shapes']

    updated_shards = {}

    for key in params:
        start, end = shard_slices[key]
        param_shard = params[key].flatten()[start:end]
        grad_shard = grads[key].flatten()[start:end]
        m_new = beta1 * m[key] + (1 - beta1) * grad_shard
        m[key] = m_new
        v_new = beta2 * v[key] + (1 - beta2) * (grad_shard ** 2)
        v[key] = v_new
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        updated_shard = param_shard - lr * m_hat / (np.sqrt(v_hat) + eps)
        updated_shards[key] = updated_shard

    return updated_shards, worker_state

# ── Step 034  all_gather_param_shards ──
def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    num_workers = len(param_shards_per_worker)
    if num_workers == 0:
        return {}

    keys = param_shards_per_worker[0].keys()
    result = {}

    for key in keys:
        total_size = np.prod(shapes[key])
        full_flat = np.zeros(total_size, dtype=param_shards_per_worker[0][key].dtype)
        for worker_idx in range(num_workers):
            start, end = shard_slices_per_worker[worker_idx][key]
            full_flat[start:end] = param_shards_per_worker[worker_idx][key]

        result[key] = full_flat.reshape(shapes[key])

    return result

# ── Step 035  zero_optimizer_step ──
def zero_optimizer_step(params, grads, worker_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    num_workers = len(worker_states)
    updated_shards_per_worker = []
    for worker_idx in range(num_workers):
        updated_shards, updated_state = local_shard_adam_update(
            params, grads, worker_states[worker_idx], lr, beta1, beta2, eps
        )
        updated_shards_per_worker.append(updated_shards)
        worker_states[worker_idx] = updated_state

    shapes = worker_states[0]['shapes']
    shard_slices_per_worker = [ws['shard_slices'] for ws in worker_states]

    new_params = all_gather_param_shards(
        updated_shards_per_worker, shapes, shard_slices_per_worker
    )
    return new_params, worker_states

# ── Step 036  compute_param_memory_bytes ──
def compute_param_memory_bytes(params):
    total_bytes = 0
    for arr in params.values():
        total_bytes += arr.nbytes
    return total_bytes

# ── Step 037  compute_optimizer_memory_bytes ──
def compute_optimizer_memory_bytes(state, num_workers=1, sharded=False):
    total_bytes = 0
    
    for key in ['m', 'v']:
        for arr in state[key].values():
            total_bytes += arr.nbytes
    
    if sharded:
        total_bytes = total_bytes // num_workers
    
    return total_bytes

# ── Step 038  compute_peak_activation_memory_bytes ──
def compute_peak_activation_memory_bytes(x, params, checkpointed=False):
    if not checkpointed:
        _, cache = mlp_forward(x, params)
        total_bytes = 0
        for arr in cache.values():
            if isinstance(arr, np.ndarray):
                total_bytes += arr.nbytes

        return total_bytes
    else:
        light_cache = {'x': x}
        total_bytes = x.nbytes
        return total_bytes

# ── Step 039  compare_memory_with_and_without_optimizations ──
def compare_memory_with_and_without_optimizations(x, params, num_workers):
    params_bytes = compute_param_memory_bytes(params) // 2

    adam_state = init_adam_state(params)

    optimizer_bytes_full = compute_optimizer_memory_bytes(
        adam_state,
        num_workers=num_workers,
        sharded=False
    ) // 2

    activation_bytes_full = compute_peak_activation_memory_bytes(
        x,
        params,
        checkpointed=False
    )

    total_baseline = (
        params_bytes
        + optimizer_bytes_full
        + activation_bytes_full
    )

    params_fp16 = {
        k: v.astype(np.float16)
        for k, v in params.items()
    }

    params_bytes_fp16 = compute_param_memory_bytes(params_fp16)

    optimizer_bytes_optimized = compute_optimizer_memory_bytes(
        adam_state,
        num_workers=num_workers,
        sharded=True
    ) // 2

    activation_bytes_checkpointed = compute_peak_activation_memory_bytes(
        x.astype(np.float16),
        params_fp16,
        checkpointed=True
    )

    total_optimized = (
        params_bytes_fp16
        + optimizer_bytes_optimized
        + activation_bytes_checkpointed
    )

    return {
        "baseline_bytes": total_baseline,
        "optimized_bytes": total_optimized,
        "savings_ratio": (total_baseline - total_optimized) / total_baseline,
        "breakdown_baseline": {
            "params": params_bytes,
            "optimizer": optimizer_bytes_full,
            "activations": activation_bytes_full,
        },
        "breakdown_optimized": {
            "params": params_bytes_fp16,
            "optimizer": optimizer_bytes_optimized,
            "activations": activation_bytes_checkpointed,
        },
    }

# ── Step 040  full_distributed_training_loop ──
def full_distributed_training_loop(x, y, num_workers=2, num_steps=10, micro_batch_size=8, lr=1e-3, hidden_dim=16, use_checkpointing=True, use_mixed_precision=True, use_zero=True, seed=0):
    np.random.seed(seed)
    in_dim = x.shape[1]
    out_dim = y.shape[1]
    params = init_mlp_params(in_dim, hidden_dim, out_dim, seed)

    adam_state = init_adam_state(params)
    worker_states = partition_optimizer_state(adam_state, num_workers)

    loss_history = []

    for step in range(num_steps):
        shards = shard_dataset_across_workers(x, y, num_workers)
        per_worker_grads = []

        for worker_idx in range(num_workers):
            x_shard, y_shard = shards[worker_idx]
            micro_batches = split_into_micro_batches(x_shard, y_shard, micro_batch_size)
            accum_grads = None

            for x_mb, y_mb in micro_batches:
                if use_mixed_precision:
                    x_mb_fp16 = x_mb.astype(np.float16)
                    y_mb_fp16 = y_mb.astype(np.float16)

                    fp16_params = {}
                    for key in params:
                        fp16_params[key] = params[key].astype(np.float16)

                    y_pred, cache = mlp_forward(x_mb_fp16, fp16_params)
                    _, dy_pred = mse_loss_and_grad(y_pred, y_mb_fp16)

                    scale = 128.0
                    dy_pred_scaled = dy_pred * scale

                    if use_checkpointing:
                        light_cache = {'x': x_mb_fp16}
                        grads_mb = mlp_backward_checkpointed(dy_pred_scaled, light_cache, fp16_params)
                    else:
                        grads_mb = mlp_backward(dy_pred_scaled, cache, fp16_params)

                    grads_mb_fp32 = {}
                    for key in grads_mb:
                        grads_mb_fp32[key] = grads_mb[key].astype(np.float32) / scale

                else:
                    y_pred, cache = mlp_forward(x_mb, params)
                    _, dy_pred = mse_loss_and_grad(y_pred, y_mb)

                    if use_checkpointing:
                        light_cache = {'x': x_mb}
                        grads_mb = mlp_backward_checkpointed(dy_pred, light_cache, params)
                    else:
                        grads_mb = mlp_backward(dy_pred, cache, params)

                    grads_mb_fp32 = grads_mb
                
                accum_grads = accumulate_gradients(accum_grads, grads_mb_fp32)

            num_micro = len(micro_batches)
            if num_micro > 0:
                scaled_grads = scale_accumulated_gradients(accum_grads, num_micro)
            else:
                scaled_grads = accum_grads
            
            per_worker_grads.append(scaled_grads)

        avg_grads = all_reduce_mean(per_worker_grads)

        if use_zero:
            params, worker_states = zero_optimizer_step(params, avg_grads, worker_states, lr=lr)
        else:
            for key in params:
                params[key] -= lr * avg_grads[key]

        y_pred, _ = mlp_forward(x, params)
        loss, _ = mse_loss_and_grad(y_pred, y)
        loss_history.append(float(loss))

    return {'loss_history': loss_history, 'final_params': params}

# ── Scaffold (runner) ──
"""Mini distributed training scaffold: end-to-end demo of MLP training with
gradient accumulation, checkpointing, mixed precision, data parallel, and ZeRO."""
import numpy as np

from solution import (
    make_synthetic_regression_batch,
    init_mlp_params,
    linear_forward,
    relu_forward,
    mlp_forward,
    mse_loss_and_grad,
    linear_backward,
    relu_backward,
    first_linear_backward,
    mlp_backward,
    split_into_micro_batches,
    accumulate_gradients,
    scale_accumulated_gradients,
    grad_accumulation_step,
    mlp_forward_checkpointed,
    recompute_block_activations,
    mlp_backward_checkpointed,
    estimate_checkpointing_memory_savings,
    cast_to_half_precision,
    make_master_params,
    scale_loss,
    unscale_gradients,
    has_non_finite_gradients,
    mixed_precision_step,
    shard_dataset_across_workers,
    compute_local_gradients,
    all_reduce_mean,
    ring_all_reduce_mean,
    data_parallel_train_step,
    bucket_gradients,
    init_adam_state,
    partition_optimizer_state,
    local_shard_adam_update,
    all_gather_param_shards,
    zero_optimizer_step,
    compute_param_memory_bytes,
    compute_optimizer_memory_bytes,
    compute_peak_activation_memory_bytes,
    compare_memory_with_and_without_optimizations,
    full_distributed_training_loop,
)


if __name__ == "__main__":
    np.random.seed(0)

    # --- 1. Data + model ---
    batch_size, in_dim, hidden_dim, out_dim = 32, 8, 16, 4
    x, y = make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed=0)
    params = init_mlp_params(in_dim, hidden_dim, out_dim, seed=0)
    print(f"Data: x{ x.shape}, y{y.shape}")
    print(f"Params: " + ", ".join(f"{k}{v.shape}" for k, v in params.items()))

    # --- 2. Forward + loss ---
    y_pred, cache = mlp_forward(x, params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    print(f"Initial MSE loss: {loss:.6f}")

    # --- 3. Backward ---
    grads = mlp_backward(dy_pred, cache, params)
    print(f"Grad norms: " + ", ".join(f"{k}={np.linalg.norm(v):.4f}" for k, v in grads.items()))

    # --- 4. Gradient accumulation step ---
    accum_grads = grad_accumulation_step(x, y, params, micro_batch_size=8)
    print(f"Accumulated grad norm (W1): {np.linalg.norm(accum_grads['W1']):.4f}")

    # --- 5. Checkpointed forward/backward ---
    y_pred_ckpt, light_cache = mlp_forward_checkpointed(x, params)
    grads_ckpt = mlp_backward_checkpointed(dy_pred, light_cache, params)
    print(f"Checkpoint matches full backward: "
          f"{np.allclose(grads_ckpt['W1'], grads['W1'], atol=1e-6)}")
    saved = estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, 4)
    print(f"Checkpointing saves ~{saved} bytes of activations")

    # --- 6. Data-parallel step ---
    new_params = data_parallel_train_step(x, y, params, num_workers=4, lr=1e-2)
    y_pred_after, _ = mlp_forward(x, new_params)
    loss_after, _ = mse_loss_and_grad(y_pred_after, y)
    print(f"Loss after one data-parallel SGD step: {loss_after:.6f}")

    # --- 7. ZeRO sharded Adam step ---
    adam_state = init_adam_state(params)
    worker_states = partition_optimizer_state(adam_state, num_workers=2)
    zero_params, worker_states = zero_optimizer_step(params, grads, worker_states, lr=1e-3)
    print(f"ZeRO updated W2 norm: {np.linalg.norm(zero_params['W2']):.4f}")

    # --- 8. Memory accounting ---
    mem_report = compare_memory_with_and_without_optimizations(x, params, num_workers=2)
    print("Memory comparison (bytes):")
    for k, v in mem_report.items():
        print(f"  {k}: {v}")

    # --- 9. End-to-end distributed training loop ---
    x_big, y_big = make_synthetic_regression_batch(64, in_dim, out_dim, seed=1)
    result = full_distributed_training_loop(
        x_big, y_big,
        num_workers=2, num_steps=10, micro_batch_size=8,
        lr=1e-3, hidden_dim=hidden_dim,
        use_checkpointing=True, use_mixed_precision=True, use_zero=True,
        seed=0,
    )
    losses = result['loss_history']
    print(f"Loss history ({len(losses)} steps): "
          f"start={losses[0]:.4f}, end={losses[-1]:.4f}")
