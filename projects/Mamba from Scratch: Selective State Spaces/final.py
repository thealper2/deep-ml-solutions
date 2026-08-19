"""
Mamba from Scratch: Selective State Spaces — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  rms_norm ──
import torch

def rms_norm(x, weight, eps=1e-5):
    """Normalize a hidden sequence with RMSNorm using a learned per-channel scale."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# ── Step 002  silu ──
def silu(x):
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)

# ── Step 003  causal_depthwise_conv1d ──
import torch
import torch.nn.functional as F

def causal_depthwise_conv1d(x, weight, bias=None):
    """Run a causal depthwise 1-D convolution over a (B, L, E) sequence.

    Args:
        x: (B, L, E) input sequence.
        weight: (E, K) per-channel kernel.
        bias: optional (E,) added after the convolution.

    Returns:
        (B, L, E) output sequence.
    """
    B, L, E = x.shape
    K = weight.shape[1]
    x_perm = x.permute(0, 2, 1)
    padding = (K - 1, 0)
    x_padded = F.pad(x_perm, padding, mode='constant', value=0)
    weight_dw = weight.unsqueeze(1)
    out = F.conv1d(x_padded, weight_dw, bias=bias, groups=E)
    out = out.permute(0 ,2, 1)
    return out

# ── Step 004  in_proj_split ──
def in_proj_split(u, weight, bias=None):
    """Project tokens to expanded inner width and split into SSM input x and gate z."""
    out = u @ weight.T

    if bias is not None:
        out = out + bias

    E = out.shape[-1] // 2
    x = out[..., :E]
    z = out[..., E:]
    return x, z

# ── Step 005  compute_delta ──
def compute_delta(x, weight, bias=None):
    """Compute a strictly positive per-token timestep Delta.

    x: (B, L, E), weight: (E, E) nn.Linear layout, bias: optional (E,).
    Returns delta of shape (B, L, E).
    """
    out = x @ weight.T

    if bias is not None:
        out = out + bias

    delta = F.softplus(out)
    return delta

# ── Step 006  project_bc ──
def project_bc(x, weight_b, weight_c):
    """Project the SSM input to input-dependent B and C state vectors of size N."""
    B_ssm = x @ weight_b.T
    C_ssm = x @ weight_c.T
    return B_ssm, C_ssm

# ── Step 007  make_diagonal_a ──
def make_diagonal_a(log_a):
    """Map unconstrained log-A of shape (E, N) to a strictly negative diagonal A."""
    return -torch.exp(log_a)

# ── Step 008  discretize_a_zoh ──
def discretize_a_zoh(delta, a):
    """Discretize a diagonal continuous state matrix with zero-order hold.

    delta: torch tensor of shape (..., d)
    a: torch tensor of shape (d, n)
    Returns a_bar of shape (..., d, n).
    """
    delta_expanded = delta.unsqueeze(-1)
    a_bar = torch.exp(delta_expanded * a)
    return a_bar

# ── Step 009  discretize_b_zoh ──
def discretize_b_zoh(delta, a, b):
    """Discretize B with the exact diagonal zero-order-hold formula.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        b_bar: (batch, seq_len, d_inner, d_state) discrete B.
    """
    delta_exp = delta.unsqueeze(-1)
    exp_delta_a = torch.exp(delta_exp * a)
    factor = (exp_delta_a - 1) / a
    b_exp = b.unsqueeze(-2)
    b_bar = factor * b_exp
    return b_bar

# ── Step 010  compare_euler_zoh_b ──
def compare_euler_zoh_b(delta, a, b):
    """Compare exact ZOH discrete B to the Euler shortcut.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        dict with keys 'b_bar_zoh', 'b_bar_euler', and 'abs_diff', each
        of shape (batch, seq_len, d_inner, d_state).
    """
    b_bar_zoh = discretize_b_zoh(delta, a, b)
    b_bar_euler = delta.unsqueeze(-1) * b.unsqueeze(2)
    abs_diff = torch.abs(b_bar_zoh - b_bar_euler)

    return {
        'b_bar_zoh': b_bar_zoh,
        'b_bar_euler': b_bar_euler,
        'abs_diff': abs_diff,
    }

# ── Step 011  siso_state_update ──
def siso_state_update(h_prev, a_bar, b_bar, c, x_t):
    """Apply one SISO state update and return the scalar readout."""
    h_t = a_bar * h_prev + b_bar * x_t
    y_t = (c * h_t).sum()
    return y_t, h_t

# ── Step 012  scan_single_channel ──
def scan_single_channel(x, a_bar, b_bar, c, h0=None):
    """Scan a single channel sequentially over time and return both the outputs and the final hidden state."""
    L = x.shape[0]
    N = a_bar.shape[1]

    if h0 is None:
        h = torch.zeros(N, dtype=x.dtype, device=x.device)
    else:
        h = h0.clone()

    y = torch.zeros(L, dtype=x.dtype, device=x.device)

    for t in range(L):
        h = a_bar[t] * h + b_bar[t] * x[t]
        y[t] = (c[t] * h).sum()

    return y, h

# ── Step 013  selective_scan ──
def selective_scan(x, a_bar, b_bar, c, h0=None):
    """Run a selective scan over a batched multi-channel sequence."""
    B, L, E = x.shape
    N = a_bar.shape[3]

    if h0 is None:
        h = torch.zeros(B, E, N, dtype=x.dtype, device=x.device)
    else:
        h = h0.clone()

    y = torch.zeros(B, L, E, dtype=x.dtype, device=x.device)

    for t in range(L):
        x_t = x[:, t, :]
        x_t_exp = x_t.unsqueeze(-1)
        h = a_bar[:, t] * h + b_bar[:, t] * x_t_exp
        y[:, t] = (c[:, t].unsqueeze(1) * h).sum(dim=-1)

    return y, h

# ── Step 014  compare_constant_vs_selective_delta ──
def compare_constant_vs_selective_delta(x, a, b, c, delta_const, delta_sel):
    """Compare SSM scan outputs under a constant Delta versus a selective Delta.

    x: (batch, seq_len, d_inner)
    a: (d_inner, d_state) strictly negative continuous diagonal A
    b: (batch, seq_len, d_state)
    c: (batch, seq_len, d_state)
    delta_const: (batch, seq_len, d_inner) non-selective timestep
    delta_sel: (batch, seq_len, d_inner) input-dependent timestep

    Returns:
        y_const: (batch, seq_len, d_inner)
        y_sel: (batch, seq_len, d_inner)
    """
    B, L, E = x.shape
    d_state = a.shape[1]
    a_bar_const = discretize_a_zoh(delta_const, a)
    b_bar_const = discretize_b_zoh(delta_const, a, b)
    a_bar_sel = discretize_a_zoh(delta_sel, a)
    b_bar_sel = discretize_b_zoh(delta_sel, a, b)
    y_const, _ = selective_scan(x, a_bar_const, b_bar_const, c)
    y_sel, _ = selective_scan(x, a_bar_sel, b_bar_sel, c)
    return y_const, y_sel

# ── Step 015  gate_scan_output ──
def gate_scan_output(y, z):
    """Modulate the selective-scan output y by the parallel gate branch z."""
    return y * silu(z)

# ── Step 016  out_proj ──
def out_proj(y, weight, bias=None):
    """Project gated scan output from d_inner back to d_model.

    y: (..., d_inner)
    weight: (d_model, d_inner)
    bias: (d_model,) or None
    Returns: (..., d_model)
    """
    out = y @ weight.T

    if bias is not None:
        out = out + bias

    return out

# ── Step 017  mamba_mixer ──
def mamba_mixer(u, params):
    """Run one full Mamba selective-SSM mixer on a token sequence.

    Args:
        u: (B, L, D) input sequence.
        params: dict of mixer weights. See the step description for keys.

    Returns:
        (B, L, D) mixer output.
    """
    B, L, D = u.shape

    in_proj_weight = params["in_proj_weight"]
    in_proj_bias = params.get("in_proj_bias", None)

    x, z = in_proj_split(u, in_proj_weight, in_proj_bias)
    E = x.shape[-1]

    conv_weight = params["conv_weight"]
    conv_bias = params.get("conv_bias", None)
    x = causal_depthwise_conv1d(x, conv_weight, conv_bias)
    x = silu(x)

    dt_weight = params["dt_weight"]
    dt_bias = params.get("dt_bias", None)
    delta = compute_delta(x, dt_weight, dt_bias)

    weight_b = params["weight_b"]
    weight_c = params["weight_c"]
    B_ssm, C_ssm = project_bc(x, weight_b, weight_c)

    log_a = params["log_a"]
    a = make_diagonal_a(log_a)

    a_bar = discretize_a_zoh(delta, a)
    b_bar = discretize_b_zoh(delta, a, B_ssm)

    y, _ = selective_scan(x, a_bar, b_bar, C_ssm)

    y = gate_scan_output(y, z)

    out_proj_weight = params["out_proj_weight"]
    out_proj_bias = params.get("out_proj_bias", None)
    out = out_proj(y, out_proj_weight, out_proj_bias)

    return out

# ── Step 018  mamba_block ──
def mamba_block(x, params):
    """Apply a pre-norm residual Mamba block to a token sequence.

    Args:
        x: (B, L, D) hidden sequence.
        params: dict with norm_weight (D,) plus every mamba_mixer key.

    Returns:
        (B, L, D) block output.
    """
    norm_weight = params["norm_weight"]
    x_norm = rms_norm(x, norm_weight)
    mixer_out = mamba_mixer(x_norm, params)
    out = x + mixer_out
    return out

# ── Step 019  run_mamba_lm_stack ──
def run_mamba_lm_stack(embeddings, params):
    """Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.

    Args:
        embeddings: (B, L, D) token embeddings.
        params: dict with key `blocks` (list of per-block dicts for `mamba_block`)
            and key `norm_weight` of shape (D,) for the final RMSNorm (eps=1e-5).

    Returns:
        (B, L, D) hidden states after the stack and final RMSNorm.
    """
    x = embeddings
    blocks = params["blocks"]

    for block_params in blocks:
        x = mamba_block(x, block_params)

    norm_weight = params["norm_weight"]
    x = rms_norm(x, norm_weight)
    return x

# ── Step 020  mamba_lm_forward ──
def mamba_lm_forward(token_ids, params):
    """Map token ids through embeddings, the Mamba stack, and an LM head.

    Args:
        token_ids: (B, L) integer tensor of token ids.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            blocks (list), and norm_weight (D,).

    Returns:
        (B, L, V) logits.
    """
    embed_weight = params["embed_weight"]
    embeddings = embed_weight[token_ids]
    hidden = run_mamba_lm_stack(embeddings, params)
    lm_head_weight = params["lm_head_weight"]
    logits = hidden @ lm_head_weight.T
    return logits

# ── Step 021  next_token_cross_entropy ──
def next_token_cross_entropy(logits, token_ids):
    """Compute the mean next-token cross-entropy from logits and token ids."""
    B, T, V = logits.shape
    logits_shifted = logits[:, :-1, :]
    targets_shifted = token_ids[:, 1:]
    logits_flat = logits_shifted.reshape(-1, V)
    targets_flat = targets_shifted.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss

# ── Step 022  sgd_training_step ──
def sgd_training_step(token_ids, params, lr):
    """Run one vanilla SGD step of next-token prediction and return the loss.

    Args:
        token_ids: (B, L) integer tensor of token ids with L >= 2.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            norm_weight (D,), and blocks (list of nested param dicts).
            Parameter tensors must have requires_grad=True and are updated in place.
        lr: vanilla SGD learning rate.

    Returns:
        Python float, the next-token cross-entropy from this step.
    """
    def zero_grads(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                zero_grads(v)
        elif isinstance(obj, list):
            for v in obj:
                zero_grads(v)
        elif isinstance(obj, torch.Tensor) and obj.requires_grad:
            if obj.grad is not None:
                obj.grad.zero_()

    zero_grads(params)

    logits = mamba_lm_forward(token_ids, params)

    loss = next_token_cross_entropy(logits, token_ids)

    loss.backward()

    def sgd_update(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                sgd_update(v)
        elif isinstance(obj, list):
            for v in obj:
                sgd_update(v)
        elif isinstance(obj, torch.Tensor) and obj.requires_grad and obj.grad is not None:
            obj.data -= lr * obj.grad.data

    sgd_update(params)

    return float(loss.item())

# ── Step 023  mamba_recurrent_step ──
def mamba_recurrent_step(token_ids, params, cache=None):
    """Consume one token and return next-token logits plus an updated SSM/conv cache."""
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(1)
    B = token_ids.shape[0]

    x = params['embed_weight'][token_ids]
    num_layers = len(params['blocks'])

    if cache is None:
        cache = {
            'conv_states': [None] * num_layers,
            'ssm_states':  [None] * num_layers,
        }

    for li, bp in enumerate(params['blocks']):
        x_norm = rms_norm(x, bp['norm_weight'])
        x_proj, z = in_proj_split(x_norm, bp['in_proj_weight'], bp.get('in_proj_bias'))
        E = x_proj.shape[-1]

        conv_weight = bp['conv_weight']
        conv_bias = bp.get('conv_bias')
        K = conv_weight.shape[1]

        conv_state = cache['conv_states'][li]
        if conv_state is None:
            conv_state = torch.zeros(B, K - 1, E, dtype=x_proj.dtype, device=x_proj.device)
        conv_input = torch.cat([conv_state, x_proj], dim=1)

        conv_out = (conv_input * conv_weight.T.unsqueeze(0)).sum(dim=1)
        if conv_bias is not None:
            conv_out = conv_out + conv_bias

        if K > 1:
            cache['conv_states'][li] = conv_input[:, 1:, :]
        else:
            cache['conv_states'][li] = torch.zeros(B, 0, E, dtype=x_proj.dtype, device=x_proj.device)

        x_conv = silu(conv_out.unsqueeze(1))

        delta = compute_delta(x_conv, bp['dt_weight'], bp.get('dt_bias'))
        B_ssm, C_ssm = project_bc(x_conv, bp['weight_b'], bp['weight_c'])
        a = make_diagonal_a(bp['log_a'])

        a_bar = discretize_a_zoh(delta, a)
        b_bar = discretize_b_zoh(delta, a, B_ssm)
        y, h_final = selective_scan(x_conv, a_bar, b_bar, C_ssm,
                                    h0=cache['ssm_states'][li])
        cache['ssm_states'][li] = h_final

        y = gate_scan_output(y, z)
        out = out_proj(y, bp['out_proj_weight'], bp.get('out_proj_bias'))
        x = x + out

    x = rms_norm(x, params['norm_weight'])
    logits = x.squeeze(1) @ params['lm_head_weight'].T
    return logits, cache

# ── Step 024  greedy_generate ──
def greedy_generate(prompt_ids, params, max_new_tokens):
    """Greedily generate new token ids from a prompt using a carried SSM cache."""
    prompt_ids = prompt_ids.clone().detach().long()
    generated = [prompt_ids]

    cache = None

    for i in range(len(prompt_ids)):
        logits, cache = mamba_recurrent_step(prompt_ids[i:i+1], params, cache)

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1)
        generated.append(next_token)
        logits, cache = mamba_recurrent_step(next_token, params, cache)

    result = torch.cat(generated, dim=0)
    return result

# ── Step 025  train_tiny_mamba_and_generate ──
import math
import torch

def train_tiny_mamba_and_generate(corpus, n_steps, lr, prompt, max_new_tokens, d_model=16, n_layers=2, d_state=4, d_inner=32, conv_kernel=3, seed=0):
    """Train a tiny character-level Mamba LM on corpus and greedily generate from prompt."""
    torch.manual_seed(seed)
    
    vocab = sorted(set(corpus))
    char_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_char = {i: c for i, c in enumerate(vocab)}
    V = len(vocab)
    
    corpus_ids = torch.tensor([char_to_id[c] for c in corpus], dtype=torch.long).unsqueeze(0)
    prompt_ids = torch.tensor([char_to_id[c] for c in prompt], dtype=torch.long)
    
    params = {}
    
    embed_weight = torch.randn(V, d_model) * 0.02
    embed_weight.requires_grad_(True)
    params['embed_weight'] = embed_weight
    
    lm_head_weight = torch.randn(V, d_model) * 0.02
    lm_head_weight.requires_grad_(True)
    params['lm_head_weight'] = lm_head_weight
    
    norm_weight = torch.ones(d_model)
    norm_weight.requires_grad_(True)
    params['norm_weight'] = norm_weight
    
    blocks = []
    for _ in range(n_layers):
        block = {}
        
        block['norm_weight'] = torch.ones(d_model)
        block['norm_weight'].requires_grad_(True)
        
        block['in_proj_weight'] = torch.randn(2 * d_inner, d_model) * 0.02
        block['in_proj_weight'].requires_grad_(True)
        block['in_proj_bias'] = torch.zeros(2 * d_inner)
        block['in_proj_bias'].requires_grad_(True)
        
        block['conv_weight'] = torch.randn(d_inner, conv_kernel) * 0.02
        block['conv_weight'].requires_grad_(True)
        block['conv_bias'] = torch.zeros(d_inner)
        block['conv_bias'].requires_grad_(True)
        
        block['dt_weight'] = torch.randn(d_inner, d_inner) * 0.02
        block['dt_weight'].requires_grad_(True)
        block['dt_bias'] = torch.zeros(d_inner)
        block['dt_bias'].requires_grad_(True)
        
        block['weight_b'] = torch.randn(d_state, d_inner) * 0.02
        block['weight_b'].requires_grad_(True)
        block['weight_c'] = torch.randn(d_state, d_inner) * 0.02
        block['weight_c'].requires_grad_(True)
        
        log_a = torch.zeros(d_inner, d_state)
        for i in range(d_inner):
            for n in range(d_state):
                log_a[i, n] = math.log(n + 1)
        log_a.requires_grad_(True)
        block['log_a'] = log_a
        
        block['out_proj_weight'] = torch.randn(d_model, d_inner) * 0.02
        block['out_proj_weight'].requires_grad_(True)
        block['out_proj_bias'] = torch.zeros(d_model)
        block['out_proj_bias'].requires_grad_(True)
        
        blocks.append(block)
    
    params['blocks'] = blocks
    
    losses = []
    for _ in range(n_steps):
        loss = sgd_training_step(corpus_ids, params, lr)
        losses.append(loss)
    
    generated_ids = greedy_generate(prompt_ids, params, max_new_tokens)
    generated_text = ''.join(id_to_char[int(id_.item())] for id_ in generated_ids)
    
    return generated_text, losses

# ── Scaffold (runner) ──
"""Tiny character-level Mamba LM: train a few steps, then greedily generate."""
import numpy as np
import torch


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    corpus = "abacabadabacaba abacabadabacaba"
    prompt = "aba"
    generated = train_tiny_mamba_and_generate(
        corpus,
        n_steps=8,
        lr=0.05,
        prompt=prompt,
        max_new_tokens=8,
        d_model=16,
        n_layers=2,
        d_state=4,
        d_inner=32,
        conv_kernel=3,
        seed=0,
    )
    print("corpus_len", len(corpus))
    print("prompt", prompt)
    print("generated", generated)


if __name__ == "__main__":
    main()
