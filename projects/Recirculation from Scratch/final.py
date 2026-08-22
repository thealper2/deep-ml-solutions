"""
Recirculation from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  rms_norm ──
def rms_norm(x, gain, eps=1e-6):
    """Apply RMSNorm over the last dimension with a learnable gain vector and eps 1e-6."""
    variance = x.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps)
    return (x * rsqrt) * gain

# ── Step 002  causal_self_attention ──
import torch
import torch.nn.functional as F
import math

def causal_self_attention(x, w_q, w_k, w_v, w_o):
    """Compute single-head causal scaled-dot-product attention."""
    B, T, D = x.shape
    Q = x @ w_q
    K = x @ w_k
    V = x @ w_v
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    out = attn @ V
    out = out @ w_o
    return out

# ── Step 003  gelu_ffn ──
import torch
import torch.nn.functional as F

def gelu_ffn(x, w_ff1, w_ff2):
    """Apply a two-layer position-wise GELU feed-forward that expands to 4D then projects back to D."""
    h = x @ w_ff1
    h = F.gelu(h)
    out = h @ w_ff2
    return out

# ── Step 004  pre_norm_block ──
def pre_norm_block(x, block):
    """Wrap attention and feed-forward as a pre-norm residual transformer block."""
    x_norm = rms_norm(x, block['attn_gain'])
    attn_out = causal_self_attention(x_norm, block['w_q'], block['w_k'], block['w_v'], block['w_o'])
    x = x + attn_out
    x_norm = rms_norm(x, block['ffn_gain'])
    ffn_out = gelu_ffn(x_norm, block['w_ff1'], block['w_ff2'])
    x = x + ffn_out
    return x

# ── Step 005  embed_tokens ──
def embed_tokens(tokens, embedding_weight):
    """Embed token ids with a (V, D) table."""
    return embedding_weight[tokens]

# ── Step 006  run_layers ──
import torch

def run_layers(x, blocks):
    """Return residual streams after every layer including the embedding as index 0."""
    residual_streams = [x]

    for block in blocks:
        x = pre_norm_block(x, block)
        residual_streams.append(x)

    return residual_streams

# ── Step 007  last_axis_l2 ──
def last_axis_l2(x):
    """Return last-axis L2 norms of x with a kept singleton dimension."""
    return torch.norm(x, dim=-1, keepdim=True)

# ── Step 008  match_source_norm ──
def match_source_norm(s, d):
    """Rescale s so its last-axis L2 matches d."""
    norm_s = last_axis_l2(s)
    norm_d = last_axis_l2(d)
    mask = norm_s > 0
    scale = torch.zeros_like(norm_s)
    scale[mask] = norm_d[mask] / norm_s[mask]
    return s * scale

# ── Step 009  convex_mix ──
def convex_mix(s, d, alpha):
    """Convex mix of destination with a magnitude-matched source."""
    matched_s = match_source_norm(s, d)
    return (1 - alpha) * d + alpha * matched_s

# ── Step 010  nonconvex_mix ──
def nonconvex_mix(s, d, alpha):
    """Nonconvex mix: destination plus a scaled matched source."""
    matched_s = match_source_norm(s, d)
    return d + alpha * matched_s

# ── Step 011  no_normalization_mix ──
def no_normalization_mix(s, d, alpha):
    """Mix source into destination with no renormalization using the raw source."""
    return (1 - alpha) * d + alpha * s

# ── Step 012  recirculate_one_position ──
def recirculate_one_position(residuals, t, source_layer, dest_layer, alpha, blocks):
    """Mix source into dest at time t then re-run blocks from dest onward."""
    new_residuals = residuals.copy()
    source = residuals[source_layer]
    dest = residuals[dest_layer]
    s_t = source[:, t:t+1, :]
    d_t = dest[:, t:t+1, :]
    mixed_t = convex_mix(s_t, d_t, alpha)
    dest_mixed = dest.clone()
    dest_mixed[:, t:t+1, :] = mixed_t
    new_residuals[dest_layer] = dest_mixed
    x = new_residuals[dest_layer]
    for i in range(dest_layer, len(blocks)):
        x = pre_norm_block(x, blocks[i])
        new_residuals[i + 1] = x
    
    return new_residuals

# ── Step 013  ramped_alpha ──
def ramped_alpha(t, alpha, ramp_steps=10):
    """Compute the ramped mixture coefficient for a 0-indexed token position t."""
    return alpha * (t / ramp_steps) if t < ramp_steps else alpha

# ── Step 014  sequential_prefill ──
def sequential_prefill(embeddings, blocks, source_layer, dest_layer, alpha, ramp_steps=10):
    """Token-by-token recirculation prefill with ramped alpha."""
    B, T, D = embeddings.shape
    x = torch.zeros_like(embeddings)
    residuals = run_layers(x, blocks)

    for t in range(T):
        prefix = embeddings[:, :t+1, :]
        prefix_residuals = run_layers(prefix, blocks)
        current_residuals = []

        for layer_idx in range(len(prefix_residuals)):
            layer_full = residuals[layer_idx]
            layer_prefix = prefix_residuals[layer_idx]
            combined = layer_full.clone()
            combined[:, t:t+1, :] = layer_prefix[:, t:t+1, :]
            current_residuals.append(combined)
        
        alpha_t = ramped_alpha(t, alpha, ramp_steps)
        current_residuals = recirculate_one_position(
            current_residuals, t, source_layer, dest_layer, alpha_t, blocks
        )
        residuals = current_residuals
    
    return residuals

# ── Step 015  insert_loop ──
def insert_loop(blocks, l1, l2):
    """Insert a looped copy of blocks from l1+1 through l2 immediately after block l2."""
    new_blocks = blocks.copy()
    loop_segment = blocks[l1+1:l2+1]
    new_blocks[l2+1:l2+1] = loop_segment
    return new_blocks

# ── Step 016  run_looped ──
def run_looped(x, blocks, l1, l2):
    """Run a looped stack and return the final residual."""
    looped_blocks = insert_loop(blocks, l1, l2)
    residuals = run_layers(x, looped_blocks)
    return residuals[-1]

# ── Step 017  tied_lm_head ──
def tied_lm_head(h, embedding_weight):
    """Project a residual stream to vocabulary logits with a tied embedding table."""
    return h @ embedding_weight.T

# ── Step 018  ntp_loss ──
import torch
import torch.nn.functional as F

def ntp_loss(logits, tokens):
    logits_shifted = logits[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    logits_flat = logits_shifted.reshape(-1, logits.shape[-1])
    tokens_flat = tokens_shifted.reshape(-1)
    loss = F.cross_entropy(logits_flat, tokens_flat, reduction='mean')
    return loss

# ── Step 019  perplexity ──
def perplexity(loss):
    """Return exp(loss) for a scalar or tensor NTP cross-entropy."""
    return torch.exp(loss) if isinstance(loss, torch.Tensor) else math.exp(loss)

# ── Step 020  concat_residuals ──
def concat_residuals(s, d):
    """Concatenate source and destination residuals along the last axis."""
    return torch.cat([s, d], dim=-1)

# ── Step 021  scalar_mix_mlp ──
import torch
import torch.nn.functional as F

def scalar_mix_mlp(concat_sd, mixer):
    """Produce scalar mixture coefficients (alpha, beta) from a concatenated residual."""
    ln_weight = mixer['ln_weight']
    ln_bias = mixer['ln_bias']
    mean = concat_sd.mean(dim=-1, keepdim=True)
    var = concat_sd.var(dim=-1, keepdim=True, unbiased=False)
    x = (concat_sd - mean) / torch.sqrt(var + 1e-5)
    x = x * ln_weight + ln_bias
    x = x @ mixer['w1'] + mixer['b1']
    x = F.gelu(x)
    x = x @ mixer['w2'] + mixer['b2']
    x = F.gelu(x)
    x = x @ mixer['w_out'] + mixer['b_out']
    x = torch.sigmoid(x)
    alpha = x[..., 0:1]
    beta = x[..., 1:2]
    return alpha, beta

# ── Step 022  vector_mix_mlp ──
def vector_mix_mlp(concat_sd, mixer):
    """Map concat(s, d) through a LayerNorm-GELU MLP to vector mixture coefficients of length D."""
    ln_weight = mixer['ln_weight']
    ln_bias = mixer['ln_bias']
    mean = concat_sd.mean(dim=-1, keepdim=True)
    var = concat_sd.var(dim=-1, keepdim=True, unbiased=False)
    x = (concat_sd - mean) / torch.sqrt(var + 1e-5)
    x = x * ln_weight + ln_bias
    x = x @ mixer['w1'] + mixer['b1']
    x = F.gelu(x)
    x = x @ mixer['w2'] + mixer['b2']
    x = F.gelu(x)
    x = x @ mixer['w_out'] + mixer['b_out']
    x = torch.sigmoid(x)
    D = x.shape[-1] // 2
    alpha = x[..., :D]
    beta = x[..., D:]
    return alpha, beta

# ── Step 023  hadamard_mix ──
def hadamard_mix(s, d, alpha, beta):
    """Hadamard mix of matched source and destination."""
    matched_s = match_source_norm(s, d)
    return alpha * matched_s + beta * d

# ── Step 024  adaptive_recirculate ──
def adaptive_recirculate(s, d, mixer):
    """Token-conditional vector mix of matched source into destination."""
    concat_sd = concat_residuals(s, d)
    alpha, beta = vector_mix_mlp(concat_sd, mixer)
    mixed = hadamard_mix(s, d, alpha, beta)
    return mixed

# ── Step 025  blockwise_recirculate ──
def blockwise_recirculate(embeddings, blocks, source_layer, dest_layer, alpha, block_size):
    """First-pass then mix K positions at a time and continue from dest."""
    B, T, D = embeddings.shape
    residuals = run_layers(embeddings, blocks)

    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        source = residuals[source_layer]
        dest = residuals[dest_layer]
        s_block = source[:, start:end, :]
        d_block = dest[:, start:end, :]
        mixed_block = convex_mix(s_block, d_block, alpha)
        dest_mixed = dest.clone()
        dest_mixed[:, start:end, :] = mixed_block
        residuals[dest_layer] = dest_mixed
        x = residuals[dest_layer]
        for i in range(dest_layer, len(blocks)):
            x = pre_norm_block(x, blocks[i])
            residuals[i + 1] = x

    return residuals

# ── Step 026  lag_diagnostic ──
def lag_diagnostic(embeddings, tokens, blocks, embedding_weight, t, k, source_layer, dest_layer, alpha):
    """Change in next-token log-likelihood at lag k after recirculating position t."""
    B, T, D = embeddings.shape
    baseline_residuals = run_layers(embeddings, blocks)
    baseline_h = baseline_residuals[-1]
    baseline_logits = tied_lm_head(baseline_h, embedding_weight)
    baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
    pred_idx = t + k
    target_idx = pred_idx + 1
    
    if pred_idx >= T - 1 or target_idx >= T:
        return torch.tensor(0.0, device=embeddings.device)
    
    baseline_ll = baseline_log_probs[:, pred_idx, :].gather(1, tokens[:, target_idx:target_idx+1])
    residuals = run_layers(embeddings, blocks)
    recirculated_residuals = recirculate_one_position(
        residuals, t, source_layer, dest_layer, alpha, blocks
    )
    recirculated_h = recirculated_residuals[-1]
    recirculated_logits = tied_lm_head(recirculated_h, embedding_weight)
    recirculated_log_probs = F.log_softmax(recirculated_logits, dim=-1)
    recirculated_ll = recirculated_log_probs[:, pred_idx, :].gather(1, tokens[:, target_idx:target_idx+1])
    delta = (recirculated_ll - baseline_ll).mean()    
    return delta

# ── Step 027  frozen_stack_adaptive_demo ──
def frozen_stack_adaptive_demo(tokens, embedding_weight, blocks, mixer, source_layer, dest_layer, alpha, steps, lr, seed=0):
    """Frozen-stack demo: baseline vs fixed recirc vs trained adaptive NTP."""
    torch.manual_seed(seed)
    
    B, T = tokens.shape
    D = embedding_weight.shape[1]
    
    embeddings = embed_tokens(tokens, embedding_weight)
    
    baseline_residuals = run_layers(embeddings, blocks)
    baseline_h = baseline_residuals[-1]
    baseline_logits = tied_lm_head(baseline_h, embedding_weight)
    baseline_loss = ntp_loss(baseline_logits, tokens)
    
    fixed_residuals = sequential_prefill(embeddings, blocks, source_layer, dest_layer, alpha)
    fixed_h = fixed_residuals[-1]
    fixed_logits = tied_lm_head(fixed_h, embedding_weight)
    fixed_loss = ntp_loss(fixed_logits, tokens)
    
    for key, value in mixer.items():
        mixer[key] = value.detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam(mixer.values(), lr=lr)
    base_residuals = run_layers(embeddings, blocks)
    
    for step in range(steps):
        s = base_residuals[source_layer].detach()
        d = base_residuals[dest_layer].detach()
        
        d_mixed = adaptive_recirculate(s, d, mixer)
        
        new_residuals = base_residuals.copy()
        new_residuals[dest_layer] = d_mixed
        
        x = new_residuals[dest_layer]
        for i in range(dest_layer, len(blocks)):
            x = pre_norm_block(x, blocks[i])
            new_residuals[i + 1] = x
        
        h = new_residuals[-1]
        logits = tied_lm_head(h, embedding_weight)
        loss = ntp_loss(logits, tokens)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        base_residuals = [r.detach() for r in new_residuals]
    
    residuals = run_layers(embeddings, blocks)
    
    s = residuals[source_layer]
    d = residuals[dest_layer]
    d_mixed = adaptive_recirculate(s, d, mixer)
    
    new_residuals = residuals.copy()
    new_residuals[dest_layer] = d_mixed
    
    x = new_residuals[dest_layer]
    for i in range(dest_layer, len(blocks)):
        x = pre_norm_block(x, blocks[i])
        new_residuals[i + 1] = x
    
    adaptive_h = new_residuals[-1]
    adaptive_logits = tied_lm_head(adaptive_h, embedding_weight)
    adaptive_loss = ntp_loss(adaptive_logits, tokens)
    
    return baseline_loss, fixed_loss, adaptive_loss

# ── Scaffold (runner) ──
"""Toy recirculation: sequential prefill vs looping vs a frozen-stack adaptive mixer."""
import torch


def _block(D):
    return {
        "attn_gain": torch.ones(D),
        "ffn_gain": torch.ones(D),
        "w_q": torch.zeros(D, D),
        "w_k": torch.zeros(D, D),
        "w_v": torch.zeros(D, D),
        "w_o": torch.zeros(D, D),
        "w_ff1": torch.zeros(D, 4 * D),
        "w_ff2": torch.zeros(4 * D, D),
    }


def main():
    torch.manual_seed(0)
    V, D, T = 8, 4, 6
    embedding_weight = torch.randn(V, D)
    blocks = [_block(D), _block(D)]
    tokens = torch.randint(0, V, (2, T))
    mixer = {
        "ln_weight": torch.ones(2 * D),
        "ln_bias": torch.zeros(2 * D),
        "w1": torch.zeros(2 * D, D),
        "b1": torch.zeros(D),
        "w2": torch.zeros(D, D),
        "b2": torch.zeros(D),
        "w_out": torch.zeros(D, 2 * D),
        "b_out": torch.zeros(2 * D),
    }
    base, fixed, adaptive = frozen_stack_adaptive_demo(
        tokens, embedding_weight, blocks, mixer,
        source_layer=2, dest_layer=1, alpha=0.15,
        steps=2, lr=0.05, seed=0,
    )
    print("baseline", float(base))
    print("fixed", float(fixed))
    print("adaptive", float(adaptive))
    e = embed_tokens(tokens, embedding_weight)
    looped = run_looped(e, blocks, 0, 1)
    print("looped", tuple(looped.shape))
    delta = lag_diagnostic(e, tokens, blocks, embedding_weight, 0, 1, 2, 1, 0.15)
    print("lag", float(delta))


if __name__ == "__main__":
    main()
