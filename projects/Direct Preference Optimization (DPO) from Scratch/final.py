"""
Direct Preference Optimization (DPO) from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  log_softmax ──
def log_softmax(logits, axis=-1):
    max_logits = np.max(logits, axis=axis, keepdims=True)
    log_sum_exp = max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=axis, keepdims=True))
    return logits - log_sum_exp

# ── Step 002  softmax ──
def softmax(logits, axis=-1):
    max_logits = np.max(logits, axis=axis, keepdims=True)
    exp_values = np.exp(logits - max_logits)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

# ── Step 003  gather_token_logprobs ──
def gather_token_logprobs(log_probs, token_ids):
    B, T, V = log_probs.shape
    result = np.zeros((B, T))
    for b in range(B):
        for t in range(T):
            result[b, t] = log_probs[b, t, token_ids[b, t]]

    return result

# ── Step 004  masked_sequence_logprob ──
def masked_sequence_logprob(token_logprobs, mask):
    return np.sum(token_logprobs * mask, axis=1)

# ── Step 005  init_policy_params ──
def init_policy_params(vocab_size, d_model, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    embed = rng.standard_normal((vocab_size, d_model)) * 0.02
    W_out = rng.standard_normal((d_model, vocab_size)) * 0.02
    b_out = np.zeros(vocab_size)

    return {
        'embed': embed,
        'W_out': W_out,
        'b_out': b_out,
    }

# ── Step 006  policy_token_logits ──
def policy_token_logits(params, token_ids):
    embed = params['embed']
    W_out = params['W_out']
    b_out = params['b_out']
    embeddings = embed[token_ids]
    logits = embeddings @ W_out + b_out
    return logits

# ── Step 007  policy_sequence_logprob ──
def policy_sequence_logprob(params, token_ids, mask):
    logits = policy_token_logits(params, token_ids)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    log_probs = logits - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    B, T, V = logits.shape
    token_logprobs = np.zeros((B, T))
    for b in range(B):
        for t in range(T):
            token_logprobs[b, t] = log_probs[b, t, token_ids[b, t]]

    return np.sum(token_logprobs * mask, axis=1)

# ── Step 008  sequence_logprob_grad ──
def sequence_logprob_grad(params, token_ids, mask):
    embed, W_out, b_out = params['embed'], params['W_out'], params['b_out']
    token_ids = np.asarray(token_ids)
    mask = np.asarray(mask, dtype=float)

    h = embed[token_ids]
    logits = h @ W_out + b_out

    p = softmax(logits)
    one_hot = np.zeros_like(p)
    B, T = token_ids.shape
    bi, ti = np.indices((B, T))
    one_hot[bi, ti, token_ids] = 1.0

    dlogits = (one_hot - p) * mask[..., None]

    db_out = dlogits.sum(axis=(0, 1))
    dW_out = np.einsum('btd,btv->dv', h, dlogits)
    dh = dlogits @ W_out.T

    dembed = np.zeros_like(embed)
    np.add.at(dembed, token_ids, dh)

    return {'embed': dembed, 'W_out': dW_out, 'b_out': db_out}

# ── Step 009  bradley_terry_loss ──
def bradley_terry_loss(reward_chosen, reward_rejected):
    margin = reward_chosen - reward_rejected
    loss = -np.log(1 / (1 + np.exp(-margin)))
    return -np.mean(np.log(1 / (1 + np.exp(-margin))))

# ── Step 010  reward_accuracy ──
def reward_accuracy(reward_chosen, reward_rejected):
    return np.mean(reward_chosen > reward_rejected)

# ── Step 011  build_preference_pairs ──
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    pairs = []
    N = len(prompts)
    for i in range(N):
        pairs.append({
            'prompt': prompts[i],
            'chosen_ids': chosen_ids[i].tolist(),
            'rejected_ids': rejected_ids[i].tolist(),
            'chosen_mask': chosen_mask[i].tolist(),
            'rejected_mask': rejected_mask[i].tolist(),
        })

    return pairs

# ── Step 012  sample_preference_batch ──
def sample_preference_batch(pairs, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_pairs = len(pairs)
    indices = rng.choice(n_pairs, size=batch_size, replace=(batch_size > n_pairs))
    batch = {}
    keys = ['chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask']
    if 'prompt' in pairs[0]:
        keys.append('prompt')

    for key in keys:
        batch[key] = np.array([pairs[i][key] for i in indices])

    return batch

# ── Step 013  freeze_reference_logprobs ──
def freeze_reference_logprobs(ref_params, pairs):
    out = []
    for pair in pairs:
        chosen_ids = np.array(pair['chosen_ids']).reshape(1, -1)
        chosen_mask = np.array(pair['chosen_mask']).reshape(1, -1)
        rejected_ids = np.array(pair['rejected_ids']).reshape(1, -1)
        rejected_mask = np.array(pair['rejected_mask']).reshape(1, -1)
        chosen_logprob = policy_sequence_logprob(ref_params, chosen_ids, chosen_mask)[0]
        rejected_logprob = policy_sequence_logprob(ref_params, rejected_ids, rejected_mask)[0]
        out.append({
            'chosen': chosen_logprob,
            'rejected': rejected_logprob,
        })

    return out

# ── Step 014  policy_reference_logratio ──
def policy_reference_logratio(policy_logprob, reference_logprob):
    return policy_logprob - reference_logprob

# ── Step 015  dpo_pair_margin ──
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    ratio_chosen = policy_logprob_chosen - ref_logprob_chosen
    ratio_rejected = policy_logprob_rejected - ref_logprob_rejected
    margin = ratio_chosen - ratio_rejected
    return beta * margin

# ── Step 016  dpo_loss ──
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    margin = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta
    )
    return np.mean(np.log(1 + np.exp(-margin)))

# ── Step 017  dpo_loss_grad ──
def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    chosen_ids = batch['chosen_ids']
    rejected_ids = batch['rejected_ids']
    chosen_mask = batch['chosen_mask']
    rejected_mask = batch['rejected_mask']

    policy_logprob_chosen = policy_sequence_logprob(params, chosen_ids, chosen_mask)
    policy_logprob_rejected = policy_sequence_logprob(params, rejected_ids, rejected_mask)

    ref_logprob_chosen = ref_logprobs_batch['chosen']
    ref_logprob_rejected = ref_logprobs_batch['rejected']

    margin = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta
    )
    loss = np.mean(np.log(1 + np.exp(-margin)))

    d_loss_d_margin = -1 / (1 + np.exp(margin))

    d_loss_d_log_pi_chosen = d_loss_d_margin * beta
    d_loss_d_log_pi_rejected = -d_loss_d_margin * beta

    grads = {key: np.zeros_like(params[key]) for key in params}
    B = len(margin)

    for b in range(B):
        chosen_mask_b = np.zeros_like(chosen_mask)
        chosen_mask_b[b] = chosen_mask[b]
        rejected_mask_b = np.zeros_like(rejected_mask)
        rejected_mask_b[b] = rejected_mask[b]

        grads_chosen_b = sequence_logprob_grad(params, chosen_ids, chosen_mask_b)
        grads_rejected_b = sequence_logprob_grad(params, rejected_ids, rejected_mask_b)

        w_chosen = d_loss_d_log_pi_chosen[b] / B
        w_rejected = d_loss_d_log_pi_rejected[b] / B

        for key in params:
            grads[key] += w_chosen * grads_chosen_b[key] + w_rejected * grads_rejected_b[key]

    return loss, grads

# ── Step 018  dpo_train_step ──
import numpy as np

def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    loss, grads = dpo_loss_grad(params, batch, ref_logprobs_batch, beta)
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grads[key]

    metrics = {'loss': loss}
    return updated_params, metrics

# ── Step 019  train_dpo ──
def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    trained_params = {key: value.copy() for key, value in params.items()}
    history = []
    
    n_pairs = len(pairs)
    
    if isinstance(ref_logprobs, dict) and 'chosen' in ref_logprobs and 'rejected' in ref_logprobs:
        ref_logprobs_list = []
        for i in range(len(ref_logprobs['chosen'])):
            ref_logprobs_list.append({
                'chosen': ref_logprobs['chosen'][i],
                'rejected': ref_logprobs['rejected'][i]
            })
        ref_logprobs = ref_logprobs_list
    
    for step in range(num_steps):
        indices = rng.choice(n_pairs, size=batch_size, replace=(batch_size > n_pairs))
        indices = [int(i) for i in indices]
        
        batch = {}
        keys = ['chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask']
        for key in keys:
            batch[key] = np.array([pairs[i][key] for i in indices])
        
        ref_logprobs_batch = {
            'chosen': np.array([ref_logprobs[i]['chosen'] for i in indices]),
            'rejected': np.array([ref_logprobs[i]['rejected'] for i in indices])
        }
        trained_params, metrics = dpo_train_step(trained_params, batch, ref_logprobs_batch, beta, learning_rate)
        metrics['step'] = step
        history.append(metrics)
    
    return trained_params, history

# ── Step 020  length_normalized_logprob ──
def length_normalized_logprob(seq_logprob, mask):
    sequence_lengths = np.sum(mask, axis=1)
    return seq_logprob / sequence_lengths

# ── Step 021  ipo_loss ──
def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    margin_chosen = policy_logprob_chosen - ref_logprob_chosen
    margin_rejected = policy_logprob_rejected - ref_logprob_rejected
    margin_diff = margin_chosen - margin_rejected
    target = 1.0 / (2.0 * beta)
    loss = np.mean((margin_diff - target) ** 2)
    return loss

# ── Step 022  implicit_reward ──
def implicit_reward(policy_logprob, reference_logprob, beta):
    return beta * (policy_logprob - reference_logprob)

# ── Step 023  preference_accuracy ──
def preference_accuracy(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    reward_chosen = implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)
    reward_rejected = implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)
    return np.mean(reward_chosen > reward_rejected)

# ── Step 024  kl_to_reference ──
def kl_to_reference(policy_logprob, reference_logprob):
    return np.mean(policy_logprob - reference_logprob)

# ── Step 025  reward_margin_stats ──
def reward_margin_stats(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    reward_chosen = implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)
    reward_rejected = implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)
    margins = reward_chosen - reward_rejected

    return {
        'mean_margin': np.mean(margins),
        'std_margin': np.std(margins),
        'frac_positive': np.mean(margins > 0)
    }

# ── Step 026  evaluate_dpo ──
def evaluate_dpo(params, pairs, ref_logprobs, beta):
    policy_logprob_chosen = []
    policy_logprob_rejected = []
    ref_logprob_chosen = []
    ref_logprob_rejected = []
    
    for i, pair in enumerate(pairs):
        chosen_ids = np.array(pair['chosen_ids']).reshape(1, -1)
        chosen_mask = np.array(pair['chosen_mask']).reshape(1, -1)
        rejected_ids = np.array(pair['rejected_ids']).reshape(1, -1)
        rejected_mask = np.array(pair['rejected_mask']).reshape(1, -1)
        
        pc = policy_sequence_logprob(params, chosen_ids, chosen_mask)
        pr = policy_sequence_logprob(params, rejected_ids, rejected_mask)
        
        policy_logprob_chosen.append(float(pc[0]) if hasattr(pc, '__len__') else float(pc))
        policy_logprob_rejected.append(float(pr[0]) if hasattr(pr, '__len__') else float(pr))
        
        if isinstance(ref_logprobs[i], dict):
            ref_logprob_chosen.append(float(ref_logprobs[i]['chosen']))
            ref_logprob_rejected.append(float(ref_logprobs[i]['rejected']))
        else:
            ref_logprob_chosen.append(float(ref_logprobs[i][0]) if hasattr(ref_logprobs[i], '__len__') else float(ref_logprobs[i]))
            ref_logprob_rejected.append(float(ref_logprobs[i][1]) if hasattr(ref_logprobs[i], '__len__') else 0.0)
    
    policy_logprob_chosen = np.array(policy_logprob_chosen)
    policy_logprob_rejected = np.array(policy_logprob_rejected)
    ref_logprob_chosen = np.array(ref_logprob_chosen)
    ref_logprob_rejected = np.array(ref_logprob_rejected)
    
    dpo_loss_value = dpo_loss(policy_logprob_chosen, policy_logprob_rejected,
                              ref_logprob_chosen, ref_logprob_rejected, beta)
    
    pref_acc = preference_accuracy(policy_logprob_chosen, policy_logprob_rejected,
                                   ref_logprob_chosen, ref_logprob_rejected, beta)
    
    all_policy = np.concatenate([policy_logprob_chosen, policy_logprob_rejected])
    all_ref = np.concatenate([ref_logprob_chosen, ref_logprob_rejected])
    kl = kl_to_reference(all_policy, all_ref)
    
    margin_stats = reward_margin_stats(policy_logprob_chosen, policy_logprob_rejected,
                                       ref_logprob_chosen, ref_logprob_rejected, beta)
    
    return {
        'dpo_loss': float(dpo_loss_value),
        'preference_accuracy': float(pref_acc),
        'kl_to_reference': float(kl),
        'mean_margin': float(margin_stats['mean_margin']),
        'std_margin': float(margin_stats['std_margin']),
        'frac_positive': float(margin_stats['frac_positive'])
    }

# ── Step 027  run_dpo_pipeline ──
def run_dpo_pipeline(vocab_size, d_model, prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta, learning_rate, num_steps, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    params = init_policy_params(vocab_size, d_model, rng=rng)
    pairs = build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    ref_logprobs = freeze_reference_logprobs(params, pairs)
    trained_params, history = train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng)
    eval_metrics = evaluate_dpo(trained_params, pairs, ref_logprobs, beta)

    return {
        'params': trained_params,
        'history': history,
        'eval_metrics': eval_metrics,
    }

# ── Scaffold (runner) ──
"""End-to-end demo of Direct Preference Optimization (DPO) with synthetic data.

Runs the preference-math helpers on hand-picked log-probs, then the full
pipeline on a LEARNABLE preference dataset (chosen responses use a "good" token
range, rejected use a disjoint "bad" range) so training visibly improves the
policy -- preference accuracy climbs toward 1.0 and the reward margin widens.
"""
import numpy as np


def main():
    np.random.seed(0)
    beta = 0.1

    # --- Core preference math on synthetic log-probs (no model required) ---
    pol_c = np.array([-2.0, -3.1, -1.4, -2.5])
    pol_r = np.array([-3.5, -2.0, -2.8, -1.9])
    ref_c = np.array([-2.2, -3.0, -1.6, -2.4])
    ref_r = np.array([-3.2, -2.3, -2.6, -2.1])
    print("DPO pair margins:", np.round(dpo_pair_margin(pol_c, pol_r, ref_c, ref_r, beta), 4))
    print("DPO loss:", float(np.round(dpo_loss(pol_c, pol_r, ref_c, ref_r, beta), 6)))
    print("IPO loss:", float(np.round(ipo_loss(pol_c, pol_r, ref_c, ref_r, beta), 6)))
    print("preference accuracy (toy):", float(preference_accuracy(pol_c, pol_r, ref_c, ref_r, beta)))
    print("reward margin stats (toy):", reward_margin_stats(pol_c, pol_r, ref_c, ref_r, beta))

    # --- Full pipeline on a LEARNABLE dataset ---
    rng = np.random.default_rng(0)
    vocab_size = 12
    d_model = 8
    n_pairs = 16
    seq_len = 6
    half = vocab_size // 2
    prompts = rng.integers(0, vocab_size, size=(n_pairs, 3))
    chosen_ids = rng.integers(0, half, size=(n_pairs, seq_len))            # "good" tokens
    rejected_ids = rng.integers(half, vocab_size, size=(n_pairs, seq_len))  # "bad" tokens
    chosen_mask = np.ones((n_pairs, seq_len))
    rejected_mask = np.ones((n_pairs, seq_len))

    result = run_dpo_pipeline(
        vocab_size, d_model, prompts, chosen_ids, rejected_ids,
        chosen_mask, rejected_mask,
        beta=0.1, learning_rate=0.3, num_steps=300, batch_size=8,
        rng=rng,
    )
    hist = result["history"]
    ev = result["eval_metrics"]
    print("")
    print("DPO training: loss", round(hist[0]["loss"], 4), "->", round(hist[-1]["loss"], 4))
    print("eval preference_accuracy:", round(float(ev["preference_accuracy"]), 4))
    print("eval mean reward margin :", round(float(ev["mean_margin"]), 4))
    print("eval dpo_loss           :", round(float(ev["dpo_loss"]), 4))


if __name__ == "__main__":
    main()
