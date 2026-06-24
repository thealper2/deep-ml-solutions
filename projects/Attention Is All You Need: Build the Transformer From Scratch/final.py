"""
Attention Is All You Need: Build the Transformer From Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  build_token_to_id_vocab ──
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    i = 0
    stoi = {}
    for special in specials:
        stoi[special] = i
        i += 1

    for sentence in sentences:
        for word in sentence.split():
            if word not in stoi.keys():
                stoi[word] = i
                i += 1

    return stoi

# ── Step 002  build_id_to_token_vocab ──
def build_id_to_token_vocab(token_to_id):
    return {v: k for k, v in token_to_id.items()}

# ── Step 003  encode_sentence_to_ids ──
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    result = []
    for word in sentence.split():
        idx = token_to_id.get(word, token_to_id[unk_token])
        result.append(idx)

    return result

# ── Step 004  decode_ids_to_tokens ──
def decode_ids_to_tokens(ids, id_to_token):
    return [id_to_token[i] for i in ids]

# ── Step 005  pad_id_sequence ──
def pad_id_sequence(ids, max_len, pad_id):
    return ids + [pad_id] * (max_len - len(ids)) if max_len > len(ids) else ids[:max_len]

# ── Step 006  stack_padded_sequences_to_batch ──
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    return torch.tensor(padded_sequences, dtype=torch.long)

# ── Step 007  scale_embeddings_by_sqrt_d_model ──
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings * (d_model ** 0.5)

# ── Step 008  compute_positional_div_term ──
import torch

def compute_positional_div_term(d_model):
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    return div_term

# ── Step 009  build_position_index_column ──
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

# ── Step 010  fill_even_indices_with_sin ──
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# ── Step 011  fill_odd_indices_with_cos ──
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ── Step 012  build_sinusoidal_positional_encoding ──
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    pe = torch.zeros((max_len, d_model))
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ── Step 013  add_positional_encoding_to_embeddings ──
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    combined_embeddings = embedded_batch + positional_encoding[:embedded_batch.shape[1]]
    return combined_embeddings

# ── Step 014  build_padding_mask ──
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    mask = (token_ids != pad_id).unsqueeze(1).unsqueeze(2)
    return mask

# ── Step 015  build_causal_mask ──
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return causal_mask.unsqueeze(0).unsqueeze(0)

# ── Step 016  combine_padding_and_causal_masks ──
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return padding_mask & causal_mask

# ── Step 017  compute_raw_attention_scores ──
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    return torch.matmul(query, key.transpose(-2, -1))

# ── Step 018  scale_attention_scores ──
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# ── Step 019  mask_attention_scores_with_neg_inf ──
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return torch.where(mask, scores, torch.tensor(float('-inf')))

# ── Step 020  softmax_attention_weights ──
import torch

def softmax_attention_weights(masked_scores):
    all_masked = (masked_scores == float('-inf')).all(dim=-1, keepdim=True)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.where(all_masked, torch.zeros_like(weights), weights)
    return weights

# ── Step 021  apply_attention_weights_to_values ──
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return attention_weights @ value

# ── Step 022  scaled_dot_product_attention ──
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)

    weights = softmax_attention_weights(scores)
    context = apply_attention_weights_to_values(weights, value)
    return context, weights

# ── Step 023  split_last_dim_into_heads ──
import torch

def split_last_dim_into_heads(tensor, num_heads):
    B, L, d_model = tensor.shape
    d_k = d_model // num_heads
    return tensor.reshape(B, L, num_heads, d_k)

# ── Step 024  transpose_heads_before_sequence ──
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.transpose(1, 2).contiguous()

# ── Step 025  merge_heads_back_to_model_dim ──
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    B, num_heads, L, d_k = multi_head_tensor.shape
    return multi_head_tensor.transpose(1, 2).contiguous().reshape(B, L, num_heads * d_k)

# ── Step 026  apply_linear_projection ──
def apply_linear_projection(x, weight, bias):
    z = x @ weight.T
    if bias is not None:
        z = z + bias

    return z

# ── Step 027  project_to_query_key_value ──
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    q_proj = apply_linear_projection(x, w_q, b_q)
    k_proj = apply_linear_projection(x, w_k, b_k)
    v_proj = apply_linear_projection(x, w_v, b_v)
    return q_proj, k_proj, v_proj

# ── Step 028  split_qkv_into_heads ──
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    batch_size, seq_len, embed_dim = q.shape
    head_dim = embed_dim // num_heads

    q_h = q.reshape(batch_size, seq_len, num_heads, head_dim)
    q_h = transpose_heads_before_sequence(q_h)

    k_h = k.reshape(batch_size, seq_len, num_heads, head_dim)
    k_h = transpose_heads_before_sequence(k_h)

    v_h = v.reshape(batch_size, seq_len, num_heads, head_dim)
    v_h = transpose_heads_before_sequence(v_h)

    return q_h, k_h, v_h

# ── Step 029  multi_head_scaled_dot_product_attention ──
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    B, num_heads, Lq, d_k = q_h.shape
    Lk = k_h.shape[2]
    d_v = v_h.shape[3]

    q_flat = q_h.reshape(B * num_heads, Lq, d_k)
    k_flat = k_h.reshape(B * num_heads, Lk, d_k)
    v_flat = v_h.reshape(B * num_heads, Lk, d_v)

    if mask is not None:
        if mask.dim() == 2:  # (B, Lk)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
        mask_flat = mask.expand(B, num_heads, Lq, Lk).reshape(B * num_heads, Lq, Lk)
    else:
        mask_flat = None

    context_flat, weights_flat = scaled_dot_product_attention(q_flat, k_flat, v_flat, mask_flat)

    context = context_flat.reshape(B, num_heads, Lq, d_v)
    weights = weights_flat.reshape(B, num_heads, Lq, Lk)
    return context, weights

# ── Step 030  merge_heads_and_project_output ──
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

# ── Step 031  assemble_multi_head_attention_forward ──
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    q_proj = apply_linear_projection(query, w_q, None)
    k_proj = apply_linear_projection(key, w_k, None)
    v_proj = apply_linear_projection(value, w_v, None)

    d_k = q_proj.shape[-1] // num_heads

    q_h = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], num_heads, d_k).transpose(1, 2)
    k_h = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], num_heads, d_k).transpose(1, 2)
    v_h = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], num_heads, d_k).transpose(1, 2)
    
    context_h, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
    
    merged = merge_heads_back_to_model_dim(context_h)
    output = apply_linear_projection(merged, w_o, None)
    
    return output

# ── Step 032  apply_ffn_first_linear_and_relu ──
def apply_ffn_first_linear_and_relu(x, w1, b1):
    hidden = x @ w1 + b1
    return torch.relu(hidden)

# ── Step 033  apply_ffn_second_linear ──
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2

# ── Step 034  position_wise_feed_forward_network ──
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(hidden, w2, b2)

# ── Step 035  compute_layer_norm_mean_and_variance ──
import torch

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return mean, var

# ── Step 036  normalize_and_scale_with_gamma_beta ──
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta

# ── Step 037  apply_residual_add_and_norm ──
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined, gamma, beta, eps)

# ── Step 038  apply_dropout_with_keep_mask ──
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return x * keep_mask / keep_prob

# ── Step 039  encoder_layer_self_attention_sublayer ──
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    attn_out = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x, attn_out, gamma, beta)

# ── Step 040  encoder_layer_feed_forward_sublayer ──
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    enc_out = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, enc_out, gamma, beta)

# ── Step 041  assemble_encoder_layer ──
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    x = encoder_layer_self_attention_sublayer(
        x,
        layer_params['w_q'],
        layer_params['w_k'],
        layer_params['w_v'],
        layer_params['w_o'],
        layer_params['attn_gamma'],
        layer_params['attn_beta'],
        num_heads,
        src_mask,
    )
    x = encoder_layer_feed_forward_sublayer(
        x,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta'],
    )
    return x

# ── Step 042  stack_encoder_layers ──
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
        
    return x

# ── Step 043  decoder_layer_masked_self_attention_sublayer ──
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    attn_out = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, attn_out, gamma, beta)

# ── Step 044  decoder_layer_cross_attention_sublayer ──
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    attn_out = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(y, attn_out, gamma, beta)

# ── Step 045  decoder_layer_feed_forward_sublayer ──
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    dec_out = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, dec_out, gamma, beta)

# ── Step 046  assemble_decoder_layer ──
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    y = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params['w_q_self'],
        layer_params['w_k_self'],
        layer_params['w_v_self'],
        layer_params['w_o_self'],
        layer_params['self_gamma'],
        layer_params['self_beta'],
        num_heads,
        tgt_mask
    )
    
    y = decoder_layer_cross_attention_sublayer(
        y,
        encoder_output,
        layer_params['w_q_cross'],
        layer_params['w_k_cross'],
        layer_params['w_v_cross'],
        layer_params['w_o_cross'],
        layer_params['cross_gamma'],
        layer_params['cross_beta'],
        num_heads,
        src_mask
    )
    
    y = decoder_layer_feed_forward_sublayer(
        y,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta']
    )
    
    return y

# ── Step 047  stack_decoder_layers ──
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)

    return y

# ── Step 048  apply_final_output_projection ──
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# ── Step 049  tie_output_projection_to_token_embeddings ──
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    return token_embedding_weight.T

# ── Step 050  apply_log_softmax_over_vocab ──
def apply_log_softmax_over_vocab(logits):
    return torch.log_softmax(logits, dim=-1)

# ── Step 051  run_transformer_forward ──
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    d_model = model_params['tgt_embedding'].shape[1]
    
    src_emb = model_params['src_embedding'][src_ids]
    tgt_emb = model_params['tgt_embedding'][tgt_ids]
    
    src_emb = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    tgt_emb = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)
    max_len = max(src_ids.shape[1], tgt_ids.shape[1])
    pos_enc = build_sinusoidal_positional_encoding(max_len, d_model)
    src_emb = add_positional_encoding_to_embeddings(src_emb, pos_enc)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, pos_enc)
    
    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, causal_mask)
    
    encoder_output = stack_encoder_layers(src_emb, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(tgt_emb, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask)
    
    logits = decoder_output @ model_params['output_projection'].T
    
    return apply_log_softmax_over_vocab(logits)

# ── Step 052  init_encoder_layer_parameters ──
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    scale = 0.02
    
    w_q = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_k = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_v = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_o = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    
    w1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * scale)
    b1 = torch.nn.Parameter(torch.zeros(d_ff))
    w2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * scale)
    b2 = torch.nn.Parameter(torch.zeros(d_model))
    
    attn_gamma = torch.nn.Parameter(torch.ones(d_model))
    attn_beta = torch.nn.Parameter(torch.zeros(d_model))
    ffn_gamma = torch.nn.Parameter(torch.ones(d_model))
    ffn_beta = torch.nn.Parameter(torch.zeros(d_model))
    
    return {
        'w_q': w_q,
        'w_k': w_k,
        'w_v': w_v,
        'w_o': w_o,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'attn_gamma': attn_gamma,
        'attn_beta': attn_beta,
        'ffn_gamma': ffn_gamma,
        'ffn_beta': ffn_beta
    }

# ── Step 053  init_decoder_layer_parameters ──
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    scale = 0.02

    w_q_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_k_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_v_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_o_self = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)

    w_q_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_k_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_v_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)
    w_o_cross = torch.nn.Parameter(torch.randn(d_model, d_model) * scale)

    w1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * scale)
    b1 = torch.nn.Parameter(torch.zeros(d_ff))
    w2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * scale)
    b2 = torch.nn.Parameter(torch.zeros(d_model))

    self_gamma = torch.nn.Parameter(torch.ones(d_model))
    self_beta = torch.nn.Parameter(torch.zeros(d_model))
    cross_gamma = torch.nn.Parameter(torch.ones(d_model))
    cross_beta = torch.nn.Parameter(torch.zeros(d_model))
    ffn_gamma = torch.nn.Parameter(torch.ones(d_model))
    ffn_beta = torch.nn.Parameter(torch.zeros(d_model))

    return {
        'w_q_self': w_q_self,
        'w_k_self': w_k_self,
        'w_v_self': w_v_self,
        'w_o_self': w_o_self,
        'w_q_cross': w_q_cross,
        'w_k_cross': w_k_cross,
        'w_v_cross': w_v_cross,
        'w_o_cross': w_o_cross,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'self_gamma': self_gamma,
        'self_beta': self_beta,
        'cross_gamma': cross_gamma,
        'cross_beta': cross_beta,
        'ffn_gamma': ffn_gamma,
        'ffn_beta': ffn_beta,
    }

# ── Step 054  init_embedding_and_projection_parameters ──
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    src_embedding = torch.nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
    tgt_embedding = torch.nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = torch.nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    return {
        'src_embedding': src_embedding,
        'tgt_embedding': tgt_embedding,
        'output_projection': output_projection,
    }

# ── Step 055  collect_model_parameters_into_list ──
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    params = []
    seen = set()

    for layer in encoder_layer_params:
        for tensor in layer.values():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                params.append(tensor)

    for layer in decoder_layer_params:
        for tensor in layer.values():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                params.append(tensor)

    for tensor in embedding_params.values():
        if id(tensor) not in seen:
            seen.add(id(tensor))
            params.append(tensor)

    return params

# ── Step 056  shift_targets_right_with_start_token ──
def shift_targets_right_with_start_token(target_ids, start_token_id):
    batch_size, seq_len = target_ids.shape
    shifted = torch.full_like(target_ids, start_token_id)
    shifted[:, 1:] = target_ids[:, :-1]
    return shifted

# ── Step 057  compute_noam_learning_rate ──
def compute_noam_learning_rate(step, d_model, warmup_steps):
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

# ── Step 058  build_uniform_smoothing_distribution ──
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    return torch.full(shape, epsilon / (vocab_size - 2), dtype=torch.float32)

# ── Step 059  set_confidence_on_gold_tokens ──
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    out = smoothed_distribution.clone()
    batch_size, seq_len, vocab_size = out.shape
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
    seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    out[batch_indices, seq_indices, gold_token_ids] = confidence
    return out

# ── Step 060  zero_pad_column_and_pad_token_rows ──
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    out = smoothed_distribution.clone()
    out[:, :, pad_id] = 0.0
    pad_mask = (gold_token_ids == pad_id)
    out[pad_mask] = 0.0
    return out

# ── Step 061  compute_label_smoothed_kl_loss ──
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    loss = torch.sum(log_probabilities * smoothed_distribution)
    return -loss if loss != 0 else loss

# ── Step 062  average_loss_over_non_pad_tokens ──
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    non_pad_mask = (gold_token_ids != pad_id)
    num_non_pad = non_pad_mask.sum()
    if num_non_pad == 0:
        return total_loss

    return total_loss / num_non_pad

# ── Step 063  compute_token_accuracy_ignoring_pad ──
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    preds = torch.argmax(log_probabilities, dim=-1)
    non_pad_mask = (gold_token_ids != pad_id)
    
    if non_pad_mask.sum() == 0:
        return torch.tensor(0.0)
    
    correct = (preds == gold_token_ids) & non_pad_mask
    return correct.sum().float() / non_pad_mask.sum().float()

# ── Step 064  initialize_adam_optimizer_state ──
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    m = []
    v = []
    for param in parameter_list:
        m.append(torch.zeros_like(param, requires_grad=False))
        v.append(torch.zeros_like(param, requires_grad=False))
        
    return {'m': m, 'v': v, 't': 0}

# ── Step 065  update_adam_first_moment ──
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    m_t = beta1 * m_prev + (1 - beta1) * grad
    return m_t

# ── Step 066  update_adam_second_moment ──
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    v_t = beta2 * v_prev + (1 - beta2) * grad ** 2
    return v_t

# ── Step 067  apply_adam_bias_correction ──
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    m_hat = m_t / (1 - beta1 ** step)
    v_hat = v_t / (1 - beta2 ** step)
    return m_hat, v_hat

# ── Step 069  apply_adam_step_to_all_parameters ──
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    t = optimizer_state['t'] + 1
    optimizer_state['t'] = t

    m_list = optimizer_state['m']
    v_list = optimizer_state['v']

    for i, param in enumerate(parameter_list):
        if param.grad is None:
            continue

        grad = param.grad

        m_new = beta1 * m_list[i] + (1 - beta1) * grad
        m_list[i] = m_new

        v_new = beta2 * v_list[i] + (1 - beta2) * (grad ** 2)
        v_list[i] = v_new

        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)

        param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    return optimizer_state

# ── Step 070  zero_all_parameter_gradients ──
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for param in parameter_list:
        if param.grad is not None:
            param.grad.detach_()
            param.grad = None

# ── Step 071  compute_batch_training_loss ──
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    smoothing = config.get('smoothing', 0.1)
    num_heads = config['num_heads']
    
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)
    
    log_probs = run_transformer_forward(src_batch, decoder_input, model_params, num_heads, pad_id)
    
    batch_size, seq_len, _ = log_probs.shape
    if smoothing == 0.0:
        smoothed_dist = torch.zeros_like(log_probs)
        smoothed_dist.scatter_(-1, tgt_batch.unsqueeze(-1), 1.0)
    else:
        uniform_val = smoothing / (vocab_size - 2)
        smoothed_dist = torch.full_like(log_probs, uniform_val)
        confidence = 1.0 - smoothing
        smoothed_dist = set_confidence_on_gold_tokens(smoothed_dist, tgt_batch, confidence)
    
    smoothed_dist = zero_pad_column_and_pad_token_rows(smoothed_dist, tgt_batch, pad_id)
    
    total_loss = compute_label_smoothed_kl_loss(log_probs, smoothed_dist)
    
    loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)
    
    return loss

# ── Step 072  run_training_step_with_backprop ──
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    zero_all_parameter_gradients(parameter_list)

    loss_tensor = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)

    loss_tensor.backward()

    d_model = config['d_model']
    warmup_steps = config['warmup_steps']
    lr = compute_noam_learning_rate(step_number, d_model, warmup_steps)

    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.98)
    epsilon = config.get('epsilon', 1e-9)

    apply_adam_step_to_all_parameters(parameter_list, optimizer_state, lr, beta1, beta2, epsilon)
    return float(loss_tensor.item())

# ── Step 073  run_training_loop_for_steps ──
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    losses = []
    num_batches = len(batches)
    
    for step in range(1, num_steps + 1):
        batch_idx = (step - 1) % num_batches
        src_batch, tgt_batch = batches[batch_idx]
        
        loss = run_training_step_with_backprop(
            src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step, config
        )
        losses.append(loss)
    
    return losses

# ── Step 074  pick_next_token_by_argmax ──
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    return torch.argmax(final_step_logits, dim=-1)

# ── Step 075  compute_length_penalty ──
def compute_length_penalty(sequence_length, alpha):
    return ((5 + sequence_length) ** alpha) / (6 ** alpha)

# ── Step 076  compute_candidate_scores ──
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    return beam_scores.unsqueeze(1) + next_token_log_probs

# ── Step 077  select_top_k_candidates ──
import torch

def select_top_k_candidates(candidate_scores, k):
    flat_scores = candidate_scores.flatten()
    top_k_values, top_k_indices = torch.topk(flat_scores, k)

    num_beams, vocab_size = candidate_scores.shape
    beam_indices = top_k_indices // vocab_size
    token_ids = top_k_indices % vocab_size

    return {
        'beam_indices': beam_indices,
        'token_ids': token_ids,
        'scores': top_k_values,
    }

# ── Step 078  append_tokens_to_beam_sequences ──
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    parent_sequences = beam_sequences[beam_indices]
    return torch.cat([parent_sequences, token_ids.unsqueeze(1)], dim=1)

# ── Step 079  mark_finished_beams ──
import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    return finished_flags | (token_ids == end_token_id)

# ── Step 080  select_best_finished_beam ──
def select_best_finished_beam(finished_sequences, finished_scores, alpha):
    best_idx = 0
    best_score = float('-inf')

    for i, seq in enumerate(finished_sequences):
        length = len(seq)
        penalty = compute_length_penalty(length, alpha)
        normalized_score = finished_scores[i] / penalty
        if normalized_score > best_score:
            best_score = normalized_score
            best_idx = i

    return {
        'sequence': finished_sequences[best_idx],
        'score': best_score,
    }

# ── Scaffold (runner) ──
"""End-to-end demo of a from-scratch Transformer: build vocab, forward pass,
a couple of training steps, and greedy/argmax next-token selection."""

import numpy as np
import torch

from solution import (
    build_token_to_id_vocab, build_id_to_token_vocab,
    encode_sentence_to_ids, decode_ids_to_tokens,
    pad_id_sequence, stack_padded_sequences_to_batch,
    scale_embeddings_by_sqrt_d_model,
    compute_positional_div_term, build_position_index_column,
    fill_even_indices_with_sin, fill_odd_indices_with_cos,
    build_sinusoidal_positional_encoding, add_positional_encoding_to_embeddings,
    build_padding_mask, build_causal_mask, combine_padding_and_causal_masks,
    compute_raw_attention_scores, scale_attention_scores,
    mask_attention_scores_with_neg_inf, softmax_attention_weights,
    apply_attention_weights_to_values, scaled_dot_product_attention,
    split_last_dim_into_heads, transpose_heads_before_sequence,
    merge_heads_back_to_model_dim, apply_linear_projection,
    project_to_query_key_value, split_qkv_into_heads,
    multi_head_scaled_dot_product_attention, merge_heads_and_project_output,
    assemble_multi_head_attention_forward,
    apply_ffn_first_linear_and_relu, apply_ffn_second_linear,
    position_wise_feed_forward_network,
    compute_layer_norm_mean_and_variance, normalize_and_scale_with_gamma_beta,
    apply_residual_add_and_norm, apply_dropout_with_keep_mask,
    encoder_layer_self_attention_sublayer, encoder_layer_feed_forward_sublayer,
    assemble_encoder_layer, stack_encoder_layers,
    decoder_layer_masked_self_attention_sublayer,
    decoder_layer_cross_attention_sublayer,
    decoder_layer_feed_forward_sublayer,
    assemble_decoder_layer, stack_decoder_layers,
    apply_final_output_projection, tie_output_projection_to_token_embeddings,
    apply_log_softmax_over_vocab, run_transformer_forward,
    init_encoder_layer_parameters, init_decoder_layer_parameters,
    init_embedding_and_projection_parameters, collect_model_parameters_into_list,
    shift_targets_right_with_start_token, compute_noam_learning_rate,
    build_uniform_smoothing_distribution, set_confidence_on_gold_tokens,
    zero_pad_column_and_pad_token_rows, compute_label_smoothed_kl_loss,
    average_loss_over_non_pad_tokens, compute_token_accuracy_ignoring_pad,
    compute_batch_training_loss, run_training_step_with_backprop,
    run_training_loop_for_steps,
    initialize_adam_optimizer_state, update_adam_first_moment,
    update_adam_second_moment, apply_adam_bias_correction,
    compute_adam_parameter_update, apply_adam_step_to_all_parameters,
    zero_all_parameter_gradients,
    pick_next_token_by_argmax, compute_length_penalty,
    compute_candidate_scores, select_top_k_candidates,
    append_tokens_to_beam_sequences, mark_finished_beams,
    select_best_finished_beam,
)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # ---- 1. Build a tiny toy parallel corpus and vocab ----
    src_sentences = ["hello world", "good morning"]
    tgt_sentences = ["bonjour monde", "bon matin"]
    all_sents = src_sentences + tgt_sentences

    tok2id = build_token_to_id_vocab(all_sents)
    id2tok = build_id_to_token_vocab(tok2id)
    pad_id = tok2id["<pad>"]
    bos_id = tok2id["<bos>"]
    eos_id = tok2id["<eos>"]
    vocab_size = len(tok2id)
    print(f"Vocab size: {vocab_size}; pad={pad_id}, bos={bos_id}, eos={eos_id}")

    # ---- 2. Encode + pad + batch ----
    max_len = 4
    src_batch = stack_padded_sequences_to_batch(
        [pad_id_sequence(encode_sentence_to_ids(s, tok2id) + [eos_id], max_len, pad_id)
         for s in src_sentences])
    tgt_batch = stack_padded_sequences_to_batch(
        [pad_id_sequence([bos_id] + encode_sentence_to_ids(s, tok2id) + [eos_id], max_len, pad_id)
         for s in tgt_sentences])
    print("src_batch shape:", tuple(src_batch.shape), "tgt_batch shape:", tuple(tgt_batch.shape))

    # ---- 3. Initialize tiny model parameters ----
    d_model, num_heads, d_ff, num_layers = 16, 2, 32, 2
    enc_layers = [init_encoder_layer_parameters(d_model, num_heads, d_ff) for _ in range(num_layers)]
    dec_layers = [init_decoder_layer_parameters(d_model, num_heads, d_ff) for _ in range(num_layers)]
    emb_params = init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True)
    model_params = {
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "embeddings": emb_params,
        "token_embedding": emb_params["src_embedding"],
        "output_projection": emb_params["output_projection"],
        "d_model": d_model,
    }
    parameter_list = collect_model_parameters_into_list(enc_layers, dec_layers, emb_params)
    print(f"Total parameter tensors: {len(parameter_list)}")

    # ---- 4. One forward pass for shape sanity ----
    log_probs = run_transformer_forward(src_batch, tgt_batch, model_params, num_heads, pad_id)
    print("log_probs shape:", tuple(log_probs.shape))

    # ---- 5. A few training steps on the toy batch and watch loss decrease ----
    config = {
        "num_heads": num_heads, "pad_id": pad_id, "bos_id": bos_id,
        "start_id": bos_id,
        "d_model": d_model, "warmup_steps": 50,
        "label_smoothing": 0.1, "smoothing": 0.1,
        "vocab_size": vocab_size,
    }
    optim_state = initialize_adam_optimizer_state(parameter_list)
    batches = [(src_batch, tgt_batch)] * 6
    losses = run_training_loop_for_steps(batches, parameter_list, model_params,
                                         optim_state, num_steps=6, config=config)
    print("loss trajectory:", [round(float(l), 4) for l in losses])

    # ---- 6. Greedy next-token pick from the final-step logits ----
    with torch.no_grad():
        final_logits = run_transformer_forward(src_batch, tgt_batch, model_params, num_heads, pad_id)
    next_tok = pick_next_token_by_argmax(final_logits[:, -1, :])
    print("argmax next tokens per row:", next_tok.tolist())
    print("decoded:", [decode_ids_to_tokens([int(t)], id2tok) for t in next_tok])
