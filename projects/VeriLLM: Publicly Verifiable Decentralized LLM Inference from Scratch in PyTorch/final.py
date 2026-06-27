"""
VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  build_char_vocab ──
def build_char_vocab(corpus):
    stoi = {}
    itos = {}
    unique = set(corpus)
    s_unique = sorted(unique)
    idx = 0
    for c in s_unique:
        stoi[c] = idx
        itos[idx] = c
        idx += 1

    return {'stoi': stoi, 'itos': itos}

# ── Step 002  encode_string ──
def encode_string(text, vocab):
    return [vocab['stoi'][c] for c in text]

# ── Step 003  decode_ids ──
def decode_ids(ids, vocab):
    return ''.join(vocab['itos'][i] for i in ids)

# ── Step 004  embed_tokens ──
import torch

def embed_tokens(token_ids, token_embedding):
    """Look up token embedding vectors for a sequence of token ids.

    Args:
        token_ids: LongTensor of shape (T,).
        token_embedding: FloatTensor of shape (vocab_size, d_model).

    Returns:
        FloatTensor of shape (T, d_model).
    """
    return token_embedding[token_ids]

# ── Step 005  add_positional_embeddings ──
import torch

def add_positional_embeddings(token_embeds, pos_embedding, start_pos=0):
    """Add the positional embedding slice [start_pos : start_pos + T] to token_embeds."""
    T, _ = token_embeds.shape
    pos_slice = pos_embedding[start_pos:start_pos + T]
    return pos_slice + token_embeds

# ── Step 006  linear_projection ──
import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    z = x @ weight
    if bias is not None:
        z = z + bias

    return z

# ── Step 007  compute_attention_scores ──
def compute_attention_scores(queries, keys):
    return queries @ keys.T

# ── Step 008  scale_attention_scores ──
def scale_attention_scores(scores, d_head):
    return scores / np.sqrt(d_head)

# ── Step 009  apply_causal_mask ──
def apply_causal_mask(scores, query_offset=0):
    Tq, Tk = scores.shape
    mask = np.zeros_like(scores)
    for i in range(Tq):
        abs_pos = query_offset + i
        mask[i, :abs_pos + 1] = 1.0

    scores = np.where(mask == 1.0, scores, -np.inf)
    return scores

# ── Step 010  softmax_attention_weights ──
import numpy as np

def softmax_attention_weights(masked_scores):
    """Convert masked attention scores to a probability distribution via softmax over the last axis."""
    max_x = np.max(masked_scores, axis=-1, keepdims=True)
    exp_x = np.exp(masked_scores - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ── Step 011  weighted_value_sum ──
import numpy as np

def weighted_value_sum(attn_weights, values):
    return np.matmul(attn_weights, values)

# ── Step 012  project_qkv ──
import numpy as np

def project_qkv(x, attn_params):
    q = linear_projection(x, attn_params['Wq'], attn_params['bq'])
    k = linear_projection(x, attn_params['Wk'], attn_params['bk'])
    v = linear_projection(x, attn_params['Wv'], attn_params['bv'])
    return q, k, v

# ── Step 013  append_kv_cache ──
def append_kv_cache(kv_cache, new_k, new_v):
    if kv_cache['k'] is None:
        kv_cache['k'] = new_k
        kv_cache['v'] = new_v
    else:
        kv_cache['k'] = np.concatenate([kv_cache['k'], new_k], axis=0)
        kv_cache['v'] = np.concatenate([kv_cache['v'], new_v], axis=0)
        
    return kv_cache

# ── Step 014  scaled_dot_product_attention_with_cache ──
import numpy as np

def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    keys, values = kv_cache['k'], kv_cache['v']
    d_k = queries.shape[-1]
    scores = compute_attention_scores(queries, keys)
    scaled_scores = scale_attention_scores(scores, d_k)
    masked_scores = apply_causal_mask(scaled_scores, query_offset)
    softmax_weights = softmax_attention_weights(masked_scores)
    weighted_sum = weighted_value_sum(softmax_weights, values)
    return weighted_sum

# ── Step 015  apply_output_projection ──
def apply_output_projection(context, attn_params):
    return linear_projection(context, attn_params['Wo'], attn_params['bo'])

# ── Step 016  single_head_causal_self_attention ──
import numpy as np

def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    d_model = x.shape[-1]
    
    Wq = attn_params.get('Wq', np.zeros((d_model, d_model)))
    Wk = attn_params.get('Wk', np.zeros((d_model, d_model)))
    Wv = attn_params.get('Wv', np.zeros((d_model, d_model)))
    Wo = attn_params.get('Wo', np.zeros((d_model, d_model)))
    bq = attn_params.get('bq', None)
    bk = attn_params.get('bk', None)
    bv = attn_params.get('bv', None)
    bo = attn_params.get('bo', None)
    
    q = linear_projection(x, Wq, bq)
    k = linear_projection(x, Wk, bk)
    v = linear_projection(x, Wv, bv)
    
    kv_cache = append_kv_cache(kv_cache, k, v)
    all_k = kv_cache['k']
    all_v = kv_cache['v']
    
    d_head = Wq.shape[0]
    scores = compute_attention_scores(q, all_k)
    scores = scale_attention_scores(scores, d_head)
    scores = apply_causal_mask(scores, query_offset)
    attn_weights = softmax_attention_weights(scores)
    context = attn_weights @ all_v
    out = linear_projection(context, Wo, bo)
    
    return out, kv_cache

# ── Step 017  ffn_first_layer_gelu ──
def ffn_first_layer_gelu(x, ffn_params):
    z = linear_projection(x, ffn_params['W1'], ffn_params.get('b1', None))
    constants = np.sqrt(2.0 / np.pi)
    return 0.5 * z * (1.0 + np.tanh(constants * (z + 0.044715 * np.power(z, 3))))

# ── Step 018  ffn_second_layer ──
def ffn_second_layer(h, ffn_params):
    z = linear_projection(h, ffn_params['W2'], ffn_params.get('b2', None))
    return z

# ── Step 019  position_wise_feed_forward ──
def position_wise_feed_forward(x, ffn_params):
    h = ffn_first_layer_gelu(x, ffn_params)
    return ffn_second_layer(h, ffn_params)

# ── Step 020  compute_mean_variance ──
import numpy as np

def compute_mean_variance(x, eps=1e-5):
    """Compute per-feature mean and variance along the last axis of x."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return mean, var

# ── Step 021  layer_norm_apply ──
import numpy as np

def layer_norm_apply(x, ln_params, eps=1e-5):
    """Normalize x over its last axis and apply gamma, beta."""
    mean, variance = compute_mean_variance(x, eps)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return ln_params['gamma'] * x_norm + ln_params['beta']

# ── Step 022  residual_add_and_norm ──
import numpy as np

def residual_add_and_norm(x, sublayer_output, ln_params, eps=1e-5):
    combined = x + sublayer_output
    return layer_norm_apply(combined, ln_params, eps)

# ── Step 023  transformer_block ──
def transformer_block(x, block_params, kv_cache, query_offset=0):
    attn_out, kv_cache = single_head_causal_self_attention(
        x,
        block_params["attn"],
        kv_cache,
        query_offset,
    )
    h = residual_add_and_norm(
        x,
        attn_out,
        block_params["ln1"],
    )

    ffn_out = position_wise_feed_forward(
        h,
        block_params["ffn"],
    )
    y = residual_add_and_norm(
        h,
        ffn_out,
        block_params["ln2"],
    )

    return y, kv_cache

# ── Step 024  lm_head_logits ──
def lm_head_logits(hidden, lm_head_params):
    linear = linear_projection(
        hidden, 
        lm_head_params['W'],
        lm_head_params['b'],
    )
    return linear

# ── Step 025  greedy_next_token ──
def greedy_next_token(logits):
    if logits.ndim == 1:
        return np.argmax(logits, axis=-1)
    else:
        max_val = np.argmax(logits[-1], axis=-1)
        return max_val

# ── Step 026  run_prefill ──
def run_prefill(prompt_ids, model_params):
    """Run prefill over the prompt tokens and build the initial KV cache per layer."""
    token_embeds = embed_tokens(prompt_ids, model_params['token_embedding'])
    x = add_positional_embeddings(token_embeds, model_params['pos_embedding'], start_pos=0)
    kv_caches = []
    for block_idx, block_params in enumerate(model_params['blocks']):
        kv_cache = {'k': None, 'v': None}
        x, kv_cache = transformer_block(x, block_params, kv_cache, query_offset=0)
        kv_caches.append(kv_cache)

    hidden = layer_norm_apply(x, model_params['ln_f']) if model_params.get('ln_f') is not None else x

    return {
        'hidden': hidden,
        'kv_caches': kv_caches,
        'next_pos': len(prompt_ids),
    }

# ── Step 027  decode_step ──
def decode_step(prev_token_id, kv_caches, next_pos, model_params):
    token_embeds = embed_tokens(np.array([prev_token_id]), model_params['token_embedding'])
    
    x = add_positional_embeddings(token_embeds, model_params['pos_embedding'], start_pos=next_pos)
    
    new_kv_caches = []
    for block_idx, block_params in enumerate(model_params['blocks']):
        kv_cache = kv_caches[block_idx]
        x, kv_cache = transformer_block(x, block_params, kv_cache, query_offset=next_pos)
        new_kv_caches.append(kv_cache)
    
    hidden = layer_norm_apply(x, model_params['ln_f']) if model_params.get('ln_f') is not None else x
    
    lm_head = model_params['lm_head']
    if isinstance(lm_head, dict):
        W = lm_head.get('W', lm_head.get('weight', None))
        b = lm_head.get('b', lm_head.get('bias', None))
    else:
        W = lm_head
        b = None
    
    if W is None:
        W = np.eye(hidden.shape[-1])
    
    logits = linear_projection(hidden, W, b)
    logits = logits.flatten()
    
    next_token = greedy_next_token(logits)
    
    return {
        'next_token': next_token,
        'logits': logits,
        'kv_caches': new_kv_caches,
        'next_pos': next_pos + 1
    }

# ── Step 028  generate_with_state_log ──
def generate_with_state_log(prompt_ids, model_params, num_new_tokens):
    """Run prefill, then decode num_new_tokens tokens, logging each step's state."""
    prefill_result = run_prefill(prompt_ids, model_params)
    kv_caches = prefill_result['kv_caches']
    next_pos = prefill_result['next_pos']
    hidden = prefill_result['hidden']

    generated_tokens = []
    step_states = []

    if num_new_tokens > 0:
        last_hidden = np.asarray(hidden)[-1]
        logits = lm_head_logits(last_hidden, model_params['lm_head'])
        logits = np.asarray(logits).flatten()
        first_token = greedy_next_token(logits)

        step_states.append({
            'next_token': first_token,
            'logits': logits,
            'kv_caches': kv_caches,
            'next_pos': next_pos,
        })
        generated_tokens.append(first_token)
        prev_token_id = first_token

        for _ in range(num_new_tokens - 1):
            result = decode_step(prev_token_id, kv_caches, next_pos, model_params)

            kv_caches = result['kv_caches']
            next_pos = result['next_pos']

            step_states.append({
                'next_token': result['next_token'],
                'logits': result['logits'],
                'kv_caches': kv_caches,
                'next_pos': next_pos,
            })
            generated_tokens.append(result['next_token'])
            prev_token_id = result['next_token']

    return {
        'generated_tokens': generated_tokens,
        'step_states': step_states,
    }

# ── Step 029  hash_tensor ──
import hashlib
import numpy as np

def hash_tensor(tensor):
    """Return a 32-byte SHA-256 digest of the tensor's shape, dtype, and contents."""
    shape_str = str(tensor.shape).encode('utf-8')
    dtype_str = str(tensor.dtype).encode('utf-8')
    bytes_data = tensor.tobytes()
    combined = shape_str + b'|' + dtype_str + b'|' + bytes_data
    return hashlib.sha256(combined).digest()

# ── Step 030  commit_decode_step ──
def commit_decode_step(step_state):
    step_index = hash_tensor(np.array([step_state['step_index']], dtype=np.int64))
    input_token = hash_tensor(np.array([step_state['input_token']], dtype=np.int64))
    next_token = hash_tensor(np.array([step_state['next_token']], dtype=np.int64))
    logits = hash_tensor(step_state['logits'])
    next_pos = hash_tensor(np.array([step_state['next_pos']], dtype=np.int64))

    kv_hashes = []
    for kv_cache in step_state['kv_caches']:
        k_hash = hash_tensor(kv_cache['k'])
        v_hash = hash_tensor(kv_cache['v'])
        kv_hashes.append(k_hash + v_hash)

    kv_combined = b''.join(kv_hashes)
    kv_combined_hash = hashlib.sha256(kv_combined).digest()

    combined = step_index + input_token + next_token + logits + kv_combined_hash + next_pos
    return hashlib.sha256(combined).digest()

# ── Step 031  hash_pair ──
import hashlib

def hash_pair(left_digest, right_digest):
    """Hash two child digests into a single parent digest."""
    combined = left_digest + right_digest
    return hashlib.sha256(combined).digest()

# ── Step 032  build_merkle_level ──
def build_merkle_level(nodes):
    parents = []
    n = len(nodes)
    for i in range(0, n, 2):
        left = nodes[i]
        if i + 1 < n:
            right = nodes[i + 1]
        else:
            right = nodes[i]

        parents.append(hash_pair(left, right))

    return parents

# ── Step 033  build_merkle_tree ──
def build_merkle_tree(leaves):
    tree = [leaves]
    current_level = leaves
    while len(current_level) > 1:
        current_level = build_merkle_level(current_level)
        tree.append(current_level)

    return tree

# ── Step 034  merkle_root ──
def merkle_root(tree):
    return tree[-1][0]

# ── Step 035  merkle_inclusion_proof ──
def merkle_inclusion_proof(tree, leaf_index):
    proof = []
    idx = leaf_index

    for level in range(len(tree) - 1):
        nodes = tree[level]
        if idx % 2 == 0:
            if idx + 1 < len(nodes):
                sibling = nodes[idx + 1]
            else:
                sibling = nodes[idx]
            
            side = 'right'
            is_right = True

        else:
            sibling = nodes[idx - 1]
            is_right = False
            side = 'left'

        proof.append({'sibling': sibling, 'is_right': is_right, 'side': side})
        idx = idx // 2

    return proof

# ── Step 036  verify_merkle_inclusion_proof ──
def verify_merkle_inclusion_proof(leaf, leaf_index, proof, root):
    current = leaf
    idx = leaf_index

    for entry in proof:
        sibling = entry['sibling']

        if entry.get('is_right', entry.get('side')) == 'right' or entry.get('is_right') == True:
            combined = current + sibling
        else:
            combined = sibling + current

        current = hashlib.sha256(combined).digest()
        idx = idx // 2

    return current == root

# ── Step 037  run_prover ──
def run_prover(model_params, prompt_ids, num_steps):
    if num_steps <= 0:
        return {'output_tokens': [], 'step_states': [], 'leaves': []}

    gen = generate_with_state_log(prompt_ids, model_params, num_steps)
    output_tokens = gen['generated_tokens']
    step_states = gen['step_states']

    leaves = [commit_decode_step(state) for state in step_states]

    return {
        'output_tokens': output_tokens,
        'step_states': step_states,
        'leaves': leaves,
    }

# ── Step 038  assemble_public_transcript ──
def assemble_public_transcript(prover_result, prompt_ids):
    leaves = prover_result['leaves']
    if leaves:
        tree = build_merkle_tree(leaves)
        root = merkle_root(tree)
    else:
        tree = [[]]
        root = None

    return {
        'prompt_ids': prompt_ids.copy() if hasattr(prompt_ids, 'copy') else prompt_ids[:],
        'output_tokens': prover_result['output_tokens'].copy() if hasattr(prover_result['output_tokens'], 'copy') else prover_result['output_tokens'][:],
        'leaves': prover_result['leaves'].copy() if hasattr(prover_result['leaves'], 'copy') else prover_result['leaves'][:],
        'tree': tree,
        'root': root,
        'step_states': prover_result['step_states'].copy() if hasattr(prover_result['step_states'], 'copy') else prover_result['step_states'][:]
    }

# ── Step 039  sample_audit_positions ──
def sample_audit_positions(seed, num_steps, k):
    if num_steps == 0 or k == 0:
        return []
        
    if k >= num_steps:
        return list(range(num_steps))
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_steps, size=k, replace=False)
    return sorted(indices.tolist())

# ── Step 040  reexecute_audited_step ──
def reexecute_audited_step(model_params, prior_kv_cache, prior_token):
    next_pos = prior_kv_cache[0]['k'].shape[0] if prior_kv_cache else 0

    result = decode_step(prior_token, prior_kv_cache, next_pos, model_params)

    return {
        'hidden': result.get('hidden'),
        'logits': result['logits'],
        'token': int(result['next_token']),
        'kv_cache_after': result['kv_caches'],
    }

# ── Step 041  recompute_step_commitment ──
def recompute_step_commitment(reexec_state, prior_kv_cache):
    step_state = {
        'step_index': reexec_state.get('step_index', 0),
        'input_token': reexec_state.get('input_token', 0),
        'next_token': reexec_state.get('token', reexec_state.get('next_token', 0)),
        'logits': reexec_state.get('logits', np.array([])),
        'kv_caches': reexec_state.get('kv_cache_after', reexec_state.get('kv_caches', [])),
        'next_pos': reexec_state.get('next_pos', 0)
    }
    
    return commit_decode_step(step_state)

# ── Step 042  check_commitment_against_proof ──
def check_commitment_against_proof(recomputed_leaf, leaf_index, proof, root):
    return verify_merkle_inclusion_proof(recomputed_leaf, leaf_index, proof, root)

# ── Step 043  check_token_matches_claim ──
def check_token_matches_claim(recomputed_token, claimed_token):
    return recomputed_token == claimed_token

# ── Step 044  run_spot_check_verification ──
def run_spot_check_verification(transcript, model_params, seed, k):
    """Run end-to-end spot-check verification of a prover transcript.

    Returns a dict with keys 'accept', 'audited_positions', 'per_audit'.
    """
    num_steps = len(transcript['output_tokens'])

    if num_steps == 0:
        return {'accept': True, 'audited_positions': [], 'per_audit': []}

    if 'tree' not in transcript:
        tree = build_merkle_tree(transcript['leaves'])
        transcript = dict(transcript)
        transcript['tree'] = tree
        if 'root' not in transcript:
            transcript['root'] = merkle_root(tree)

    audited_positions = sample_audit_positions(seed, num_steps, k)

    per_audit = []
    all_passed = True

    for pos in audited_positions:
        step_state = transcript['step_states'][pos]

        if pos == 0:
            prior_kv_cache = step_state['kv_caches']
            prior_token = step_state['input_token']
        else:
            prev_state = transcript['step_states'][pos - 1]
            prior_kv_cache = prev_state['kv_caches']
            prior_token = prev_state['next_token']

        reexec_result = reexecute_audited_step(model_params, prior_kv_cache, prior_token)
        recomputed_leaf = recompute_step_commitment(reexec_result, prior_kv_cache)

        leaf_index = pos
        proof = merkle_inclusion_proof(transcript['tree'], leaf_index)
        commitment_ok = check_commitment_against_proof(recomputed_leaf, leaf_index, proof, transcript['root'])

        claimed_token = transcript['output_tokens'][pos]
        token_ok = check_token_matches_claim(reexec_result['token'], claimed_token)

        per_audit.append({
            'commitment_ok': commitment_ok,
            'token_ok': token_ok,
        })

        if not commitment_ok or not token_ok:
            all_passed = False

    return {
        'accept': all_passed,
        'audited_positions': audited_positions,
        'per_audit': per_audit,
    }

# ── Step 045  tamper_transcript_flip_token ──
def tamper_transcript_flip_token(transcript, position, new_token):
    new_transcript = dict(transcript)
    output_tokens = transcript['output_tokens']
    new_transcript['output_tokens'] = (
        output_tokens.copy() if hasattr(output_tokens, 'copy') else output_tokens[:]
    )
    new_transcript['output_tokens'][position] = new_token
    return new_transcript

# ── Step 046  detection_probability ──
import math

def detection_probability(num_steps, num_corrupted, k):
    if k == 0 or num_corrupted == 0:
        return 0.0

    if num_corrupted >= num_steps:
        return 1.0

    if k >= num_steps:
        return 1.0 if num_corrupted > 0 else 0.0

    clean = num_steps - num_corrupted
    if k > clean:
        return 1.0

    miss_prob = math.comb(clean, k) / math.comb(num_steps, k)
    return 1.0 - miss_prob

# ── Step 047  verifier_cost_fraction ──
def verifier_cost_fraction(num_steps, k):
    return k / num_steps

# ── Step 048  show_tampered_transcript_rejected ──
def show_tampered_transcript_rejected(transcript, model_params, position, new_token, seed, k):
    tampered = tamper_transcript_flip_token(transcript, position, new_token)
    result = run_spot_check_verification(tampered, model_params, seed, k)
    
    return {
        'tampered_transcript': tampered,
        'result': result,
        'rejected': not result['accept']
    }

# ── Step 049  sample_verifier_committee ──
import random

def sample_verifier_committee(verifier_ids, committee_size, seed):
    if committee_size <= 0:
        return []
        
    if committee_size >= len(verifier_ids):
        return verifier_ids.copy()
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(verifier_ids), size=committee_size, replace=False)
    return [verifier_ids[i] for i in indices]

# ── Step 050  collect_verifier_votes ──
def collect_verifier_votes(committee, transcript, model_params, k, base_seed):
    votes = []

    for verifier_id in committee:
        seed = hash(verifier_id) ^ base_seed
        if seed < 0:
            seed = -seed

        seed = seed % (2 ** 31)

        result = run_spot_check_verification(transcript, model_params, seed, k)

        votes.append({
            'verifier_id': verifier_id,
            'vote': result['accept'],
            'result': result,
        })

    return votes

# ── Step 051  aggregate_votes_majority ──
def aggregate_votes_majority(votes):
    accept_count = sum(1 for v in votes if v['vote'])
    reject_count = len(votes) - accept_count
    verdict = accept_count > reject_count

    return {
        'verdict': verdict,
        'accept_count': accept_count,
        'reject_count': reject_count,
    }

# ── Step 052  reward_honest_participants ──
def reward_honest_participants(balances, worker_id, votes, verdict, reward_worker, reward_verifier):
    new_balances = balances.copy()

    if verdict:
        new_balances[worker_id] = new_balances.get(worker_id, 0.0) + reward_worker

    for vote in votes:
        verifier_id = vote['verifier_id']
        if vote['vote'] == verdict:
            new_balances[verifier_id] = new_balances.get(verifier_id, 0.0) + reward_verifier

    return new_balances

# ── Step 053  slash_worker ──
def slash_worker(balances, worker_id, slash_amount):
    new_balances = balances.copy()
    current = new_balances.get(worker_id, 0.0)
    new_balances[worker_id] = current - slash_amount
    return new_balances

# ── Step 054  assign_dual_role ──
def assign_dual_role(node_ids, worker_id, committee_size, seed):
    if worker_id not in node_ids:
        node_ids = node_ids + [worker_id]

    other_ids = [n for n in node_ids if n != worker_id]

    if committee_size == 1:
        return {'worker_id': worker_id, 'committee': [worker_id]}

    if len(other_ids) < committee_size - 1:
        committee = other_ids.copy()
        while len(committee) < committee_size:
            committee.append(worker_id)

        return {'worker_id': worker_id, 'committee': committee}

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(other_ids), size=committee_size - 1, replace=False)
    committee = [other_ids[i] for i in indices]
    committee.append(worker_id)

    rng.shuffle(committee)

    return {'worker_id': worker_id, 'committee': committee}

# ── Step 055  run_honest_round ──
def run_honest_round(model_params, prompt_ids, num_steps, verifier_ids, worker_id, committee_size, k, seed, balances, reward_worker, reward_verifier):
    prover_result = run_prover(model_params, prompt_ids, num_steps)

    transcript = assemble_public_transcript(prover_result, prompt_ids)

    committee = sample_verifier_committee(verifier_ids, committee_size, seed)

    votes = collect_verifier_votes(committee, transcript, model_params, k, seed)

    agg = aggregate_votes_majority(votes)
    verdict = agg['verdict']

    new_balances = reward_honest_participants(
        balances, worker_id, votes, verdict, reward_worker, reward_verifier
    )

    return {
        'transcript': transcript,
        'votes': votes,
        'verdict': verdict,
        'balances': new_balances,
    }

# ── Step 056  run_malicious_round ──
def run_malicious_round(model_params, prompt_ids, num_steps, verifier_ids, worker_id, committee_size, k, seed, balances, slash_amount, tamper_position, new_token):
    gen = generate_with_state_log(prompt_ids, model_params, num_steps)
    step_states = gen['step_states']
    output_tokens = gen['generated_tokens']

    prev_token = prompt_ids[-1]
    for i, state in enumerate(step_states):
        state['step_index'] = i
        state['input_token'] = prev_token
        prev_token = state['next_token']

    leaves = [commit_decode_step(state) for state in step_states]
    prover_result = {
        'output_tokens': output_tokens,
        'step_states': step_states,
        'leaves': leaves,
    }
    transcript = assemble_public_transcript(prover_result, prompt_ids)

    tampered_transcript = tamper_transcript_flip_token(transcript, tamper_position, new_token)

    committee = sample_verifier_committee(verifier_ids, committee_size, seed)

    votes = collect_verifier_votes(committee, tampered_transcript, model_params, k, seed)

    agg = aggregate_votes_majority(votes)
    verdict = agg['verdict']

    if not verdict:
        new_balances = slash_worker(balances, worker_id, slash_amount)
    else:
        new_balances = balances.copy()

    return {
        'committee': committee,
        'votes': votes,
        'aggregate': agg,
        'accept_count': agg['accept_count'],
        'reject_count': agg['reject_count'],
        'verdict': verdict,
        'balances': new_balances,
        'tampered_transcript': tampered_transcript,
    }

# ── Step 057  report_end_to_end_verification_cost ──
def report_end_to_end_verification_cost(num_steps, committee_size, k):
    per_verifier_fraction = verifier_cost_fraction(num_steps, k)
    committee_fraction = per_verifier_fraction * committee_size
    full_reexec_fraction = 1.0
    return {
        'per_verifier_fraction': per_verifier_fraction,
        'committee_fraction': committee_fraction,
        'full_reexec_fraction': full_reexec_fraction
    }

# ── Scaffold (runner) ──
"""End-to-end demo of VeriLLM: tiny from-scratch GPT-style transformer with KV cache,
Merkle-committed greedy decoding, spot-check verification, and a committee simulation."""

import numpy as np
import torch

from solution import (
    build_char_vocab, encode_string, decode_ids,
    embed_tokens, add_positional_embeddings, linear_projection,
    compute_attention_scores, scale_attention_scores, apply_causal_mask,
    softmax_attention_weights, weighted_value_sum, project_qkv,
    append_kv_cache, scaled_dot_product_attention_with_cache,
    apply_output_projection, single_head_causal_self_attention,
    ffn_first_layer_gelu, ffn_second_layer, position_wise_feed_forward,
    compute_mean_variance, layer_norm_apply, residual_add_and_norm,
    transformer_block, lm_head_logits, greedy_next_token,
    run_prefill, decode_step, generate_with_state_log,
    hash_tensor, commit_decode_step, hash_pair, build_merkle_level,
    build_merkle_tree, merkle_root, merkle_inclusion_proof,
    verify_merkle_inclusion_proof,
    run_prover, assemble_public_transcript, sample_audit_positions,
    reexecute_audited_step, recompute_step_commitment,
    check_commitment_against_proof, check_token_matches_claim,
    run_spot_check_verification, tamper_transcript_flip_token,
    detection_probability, verifier_cost_fraction,
    show_tampered_transcript_rejected,
    sample_verifier_committee, collect_verifier_votes,
    aggregate_votes_majority, reward_honest_participants, slash_worker,
    assign_dual_role, run_honest_round, run_malicious_round,
    report_end_to_end_verification_cost,
)


def make_toy_model_params(vocab_size, d_model=16, d_ff=32, n_layers=2, max_pos=64, rng=None):
    """Build a tiny random transformer parameter dict matching the solution's expected layout."""
    r = rng if rng is not None else np.random.default_rng(0)
    def randn(*shape):
        return r.standard_normal(shape).astype(np.float32) * 0.05
    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    blocks = []
    for _ in range(n_layers):
        attn = {
            "Wq": randn(d_model, d_model), "bq": zeros(d_model),
            "Wk": randn(d_model, d_model), "bk": zeros(d_model),
            "Wv": randn(d_model, d_model), "bv": zeros(d_model),
            "Wo": randn(d_model, d_model), "bo": zeros(d_model),
        }
        ffn = {
            "W1": randn(d_model, d_ff), "b1": zeros(d_ff),
            "W2": randn(d_ff, d_model), "b2": zeros(d_model),
        }
        ln1 = {"gamma": np.ones(d_model, dtype=np.float32), "beta": zeros(d_model)}
        ln2 = {"gamma": np.ones(d_model, dtype=np.float32), "beta": zeros(d_model)}
        blocks.append({"attn": attn, "ffn": ffn, "ln1": ln1, "ln2": ln2})

    return {
        "token_embedding": randn(vocab_size, d_model),
        "pos_embedding": randn(max_pos, d_model),
        "blocks": blocks,
        "ln_f": {"gamma": np.ones(d_model, dtype=np.float32), "beta": zeros(d_model)},
        "lm_head": {"W": randn(d_model, vocab_size), "b": zeros(vocab_size)},
        "d_model": d_model, "n_layers": n_layers,
    }


def _vocab_size(vocab):
    if isinstance(vocab, tuple):
        return max(len(v) for v in vocab)
    if isinstance(vocab, dict):
        for key in ("stoi", "itos"):
            if key in vocab and hasattr(vocab[key], "__len__"):
                return len(vocab[key])
        return len(vocab)
    return len(vocab)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Data prep ---
    corpus = "hello verifiable world of decentralized llm inference"
    vocab = build_char_vocab(corpus)
    vocab_size = _vocab_size(vocab)
    prompt_ids = encode_string("hello ", vocab)
    print(f"vocab size = {vocab_size} | prompt ids = {prompt_ids}")
    print(f"round-trip decode = {decode_ids(prompt_ids, vocab)!r}")

    # --- Build tiny model ---
    model_params = make_toy_model_params(vocab_size=vocab_size)
    num_steps = 6

    # --- Prover: greedy decode with per-step Merkle commitments ---
    prover_result = run_prover(model_params, prompt_ids, num_steps=num_steps)
    transcript = assemble_public_transcript(prover_result, prompt_ids)
    print(f"output tokens   = {transcript['output_tokens']}")
    print(f"merkle root[:8] = {str(transcript['root'])[:16]}...")

    # --- Honest spot-check verification ---
    k = 2
    accepted = run_spot_check_verification(transcript, model_params, seed=42, k=k)
    print(f"honest spot-check accepted? {accepted}")
    print(f"verifier cost fraction (k/N) = {verifier_cost_fraction(num_steps, k):.3f}")
    print(f"detection prob (1 corrupted) = {detection_probability(num_steps, 1, k):.3f}")

    # --- Tamper a token and show rejection ---
    tamper_pos = 1
    new_token = (transcript["output_tokens"][tamper_pos] + 1) % vocab_size
    rejected = show_tampered_transcript_rejected(
        transcript, model_params, position=tamper_pos,
        new_token=new_token, seed=42, k=num_steps,
    )
    print(f"tampered transcript rejected? {rejected}")

    # --- Committee simulation: honest round ---
    # Use integer verifier ids since collect_verifier_votes calls int(verifier_id).
    verifier_ids = list(range(5))
    worker_id = "w0"
    balances = {worker_id: 0, **{v: 0 for v in verifier_ids}}
    committee_size, audit_k = 3, 2

    honest_result = run_honest_round(
        model_params, prompt_ids, num_steps, verifier_ids, worker_id,
        committee_size=committee_size, k=audit_k, seed=7,
        balances=dict(balances), reward_worker=10, reward_verifier=1,
    )
    honest_balances = honest_result['balances']
    honest_verdict = honest_result['verdict']
    print(f"honest verdict = {honest_verdict} | balances = {honest_balances}")

    # --- Committee simulation: malicious round ---
    mal_result = run_malicious_round(
        model_params, prompt_ids, num_steps, verifier_ids, worker_id,
        committee_size=committee_size, k=num_steps, seed=7,
        balances=dict(balances), slash_amount=20,
        tamper_position=2,
        new_token=(transcript["output_tokens"][2] + 1) % vocab_size,
    )
    mal_balances = mal_result['balances']
    mal_verdict = mal_result['verdict']
    print(f"malicious verdict = {mal_verdict} | balances = {mal_balances}")

    cost = report_end_to_end_verification_cost(num_steps, committee_size, audit_k)
    print(f"end-to-end verifier cost vs full re-exec baseline 1.0: {cost['committee_fraction']:.3f}")
