"""
NextLat from Scratch: Next-Latent Prediction in PyTorch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  grid_step ──
def grid_step(pos: tuple, action: int, G: int) -> tuple:
    row, col = pos
    
    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        col -= 1
    elif action == 3:
        col += 1
    
    if 0 <= row < G and 0 <= col < G:
        return (row, col), True
        
    return pos, False

# ── Step 002  legal_actions ──
def legal_actions(pos: tuple, G: int) -> list:
    actions = []
    for action in range(4):
        _, legal = grid_step(pos, action, G)
        if legal:
            actions.append(action)

    return actions

# ── Step 003  random_walk_to_goal ──
def random_walk_to_goal(start: tuple, goal: tuple, G: int, max_len: int, rng) -> list:
    pos = start
    moves = []

    for _ in range(max_len):
        if pos == goal:
            break

        legal = legal_actions(pos, G)
        action = rng.choice(legal)
        pos, _ = grid_step(pos, action, G)
        moves.append(int(action))

    return moves

# ── Step 004  encode_sequence ──
import torch

def encode_sequence(start: tuple, goal: tuple, moves: list, G: int, T: int) -> tuple:
    start_token = 4 + start[0] * G + start[1]
    goal_token = 4 + goal[0] * G + goal[1]
    EOS = 4 + G * G
    
    max_moves = T - 3
    if len(moves) > max_moves:
        moves = moves[:max_moves]
    
    tokens_list = [start_token, goal_token] + moves + [EOS]
    real_len = len(tokens_list)
    
    while len(tokens_list) < T:
        tokens_list.append(EOS)
    
    tokens = torch.tensor(tokens_list, dtype=torch.long)
    
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:real_len] = True
    
    return tokens, mask

# ── Step 005  make_dataset ──
import torch
import numpy as np

def make_dataset(n: int, G: int, T: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    
    tokens_list = []
    masks_list = []
    states_list = []
    
    for _ in range(n):
        start_row = rng.integers(0, G)
        start_col = rng.integers(0, G)
        goal_row = rng.integers(0, G)
        goal_col = rng.integers(0, G)
        
        start = (start_row, start_col)
        goal = (goal_row, goal_col)
        
        max_len = T - 3
        moves = random_walk_to_goal(start, goal, G, max_len, rng)
        
        tokens, mask = encode_sequence(start, goal, moves, G, T)
        
        pos = start
        states = torch.zeros(T, dtype=torch.long)
        
        states[0] = start[0] * G + start[1]
        
        states[1] = start[0] * G + start[1]
        
        for i, move in enumerate(moves):
            pos, _ = grid_step(pos, move, G)
            states[i + 2] = pos[0] * G + pos[1]
        
        last_pos_idx = pos[0] * G + pos[1]
        for t in range(len(moves) + 2, T):
            states[t] = last_pos_idx
        
        tokens_list.append(tokens)
        masks_list.append(mask)
        states_list.append(states)
    
    return {
        'tokens': torch.stack(tokens_list),
        'mask': torch.stack(masks_list),
        'states': torch.stack(states_list),
        'G': G
    }

# ── Step 006  get_batch ──
def get_batch(dataset: dict, batch_size: int, step: int) -> dict:
    n = dataset['tokens'].shape[0]
    T = dataset['tokens'].shape[1]

    indices = [(step * batch_size + i) % n for i in range(batch_size)]

    tokens = dataset['tokens'][indices]
    mask = dataset['mask'][indices]
    states = dataset['states'][indices]

    x = tokens[:, :-1]
    y = tokens[:, 1:]
    mask_shifted = mask[:, 1:]
    states_shifted = states[:, :-1]

    return {
        'x': x,
        'y': y,
        'mask': mask_shifted,
        'states': states_shifted,
    }

# ── Step 007  causal_mask ──
import torch

def causal_mask(T: int):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))

# ── Step 008  init_gpt_params ──
import torch

def init_gpt_params(vocab_size: int, d_model: int, n_layers: int, max_len: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    std = 0.02
    
    params = {}
    
    params['wte'] = torch.randn(vocab_size, d_model) * std
    params['wte'].requires_grad_(True)
    
    params['wpe'] = torch.randn(max_len, d_model) * std
    params['wpe'].requires_grad_(True)
    
    for l in range(n_layers):
        params[f'ln1_w{l}'] = torch.ones(d_model)
        params[f'ln1_w{l}'].requires_grad_(True)
        params[f'ln1_b{l}'] = torch.zeros(d_model)
        params[f'ln1_b{l}'].requires_grad_(True)
        
        params[f'qkv_w{l}'] = torch.randn(d_model, 3 * d_model) * std
        params[f'qkv_w{l}'].requires_grad_(True)
        params[f'qkv_b{l}'] = torch.zeros(3 * d_model)
        params[f'qkv_b{l}'].requires_grad_(True)
        
        params[f'proj_w{l}'] = torch.randn(d_model, d_model) * std
        params[f'proj_w{l}'].requires_grad_(True)
        params[f'proj_b{l}'] = torch.zeros(d_model)
        params[f'proj_b{l}'].requires_grad_(True)
        
        params[f'ln2_w{l}'] = torch.ones(d_model)
        params[f'ln2_w{l}'].requires_grad_(True)
        params[f'ln2_b{l}'] = torch.zeros(d_model)
        params[f'ln2_b{l}'].requires_grad_(True)
        
        params[f'fc_w{l}'] = torch.randn(d_model, 4 * d_model) * std
        params[f'fc_w{l}'].requires_grad_(True)
        params[f'fc_b{l}'] = torch.zeros(4 * d_model)
        params[f'fc_b{l}'].requires_grad_(True)
        
        params[f'fc2_w{l}'] = torch.randn(4 * d_model, d_model) * std
        params[f'fc2_w{l}'].requires_grad_(True)
        params[f'fc2_b{l}'] = torch.zeros(d_model)
        params[f'fc2_b{l}'].requires_grad_(True)
    
    params['lnf_w'] = torch.ones(d_model)
    params['lnf_w'].requires_grad_(True)
    params['lnf_b'] = torch.zeros(d_model)
    params['lnf_b'].requires_grad_(True)
    
    params['head_w'] = torch.randn(d_model, vocab_size) * std
    params['head_w'].requires_grad_(True)
    params['head_b'] = torch.zeros(vocab_size)
    params['head_b'].requires_grad_(True)
    
    return params

# ── Step 009  attention_block ──
import torch
import torch.nn.functional as F

def attention_block(x, params: dict, layer: int, n_heads: int):
    ln_w = params[f'ln1_w{layer}']
    ln_b = params[f'ln1_b{layer}']
    eps = 1e-5
    
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    z = (x - mean) / torch.sqrt(var + eps)
    z = z * ln_w + ln_b
    
    qkv_w = params[f'qkv_w{layer}']
    qkv_b = params[f'qkv_b{layer}']
    qkv = z @ qkv_w + qkv_b
    
    B, T, _ = qkv.shape
    d = x.shape[-1]
    head_dim = d // n_heads
    
    q, k, v = torch.split(qkv, d, dim=-1)
    
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    mask = causal_mask(T)
    mask = mask.to(x.device)
    scores = scores.masked_fill(~mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    
    out = out.transpose(1, 2).contiguous().view(B, T, d)
    
    proj_w = params[f'proj_w{layer}']
    proj_b = params[f'proj_b{layer}']
    out = out @ proj_w + proj_b
    
    return x + out

# ── Step 010  mlp_block ──
import torch
import torch.nn.functional as F

def mlp_block(x, params: dict, layer: int):
    ln_w = params[f'ln2_w{layer}']
    ln_b = params[f'ln2_b{layer}']
    eps = 1e-5

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    z = (x - mean) / torch.sqrt(var + eps)
    z = z * ln_w + ln_b

    fc_w = params[f'fc_w{layer}']
    fc_b = params[f'fc_b{layer}']
    h = z @ fc_w + fc_b

    h = F.gelu(h, approximate='tanh')

    fc2_w = params[f'fc2_w{layer}']
    fc2_b = params[f'fc2_b{layer}']
    out = h @ fc2_w + fc2_b
    return x + out

# ── Step 011  gpt_hidden_states ──
import torch
import torch.nn.functional as F

def gpt_hidden_states(tokens, params: dict, n_heads: int):
    n_layers = 0
    while f'ln1_w{n_layers}' in params:
        n_layers += 1

    wte = params['wte']
    wpe = params['wpe']
    B, T = tokens.shape

    x = wte[tokens] + wpe[:T]

    for layer in range(n_layers):
        x = attention_block(x, params, layer, n_heads)
        x = mlp_block(x, params, layer)

    lnf_w = params['lnf_w']
    lnf_b = params['lnf_b']
    eps = 1e-5

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x * lnf_w + lnf_b
    return x

# ── Step 012  output_head ──
def output_head(h, params: dict):
    head_w = params['head_w']
    head_b = params['head_b']
    return h @ head_w + head_b

# ── Step 013  next_token_loss ──
import torch
import torch.nn.functional as F

def next_token_loss(logits, targets, mask):
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    masked_ce = ce[mask_flat]
    if masked_ce.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    return masked_ce.mean()

# ── Step 014  init_dynamics_params ──
import torch

def init_dynamics_params(d_model: int, hidden: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    std = 0.02

    params = {}

    params['W1'] = torch.randn(2 * d_model, hidden) * std
    params['W1'].requires_grad_(True)

    params['b1'] = torch.zeros(hidden)
    params['b1'].requires_grad_(True)

    params['W2'] = torch.randn(hidden, hidden) * std
    params['W2'].requires_grad_(True)

    params['b2'] = torch.zeros(hidden)
    params['b2'].requires_grad_(True)

    params['W3'] = torch.randn(hidden, d_model) * std
    params['W3'].requires_grad_(True)

    params['b3'] = torch.zeros(d_model)
    params['b3'].requires_grad_(True)

    return params

# ── Step 015  latent_transition ──
import torch
import torch.nn.functional as F

def latent_transition(h, x_emb, dyn: dict):
    z = torch.cat([h, x_emb], dim=-1)

    eps = 1e-5
    mean = z.mean(dim=-1, keepdim=True)
    var = z.var(dim=-1, keepdim=True, unbiased=False)
    z_norm = (z - mean) / torch.sqrt(var + eps)

    a1 = F.gelu(z_norm @ dyn['W1'] + dyn['b1'], approximate='tanh')

    a2 = F.gelu(a1 @ dyn['W2'] + dyn['b2'], approximate='tanh')

    delta = a2 @ dyn['W3'] + dyn['b3']

    return delta + h

# ── Step 016  rollout_latents ──
def rollout_latents(h, x, params: dict, dyn: dict, d_steps: int) -> list:
    B, T, d = h.shape
    wte = params['wte']
    
    h_hat = h[:, :T - d_steps]
    predictions = []
    
    for i in range(1, d_steps + 1):
        token_indices = x[:, i:T - d_steps + i]
        emb = wte[token_indices]
        
        h_hat = latent_transition(h_hat, emb, dyn)
        predictions.append(h_hat)
    
    return predictions

# ── Step 017  next_hidden_loss ──
import torch
import torch.nn.functional as F

def next_hidden_loss(h, h_hats: list, mask, beta: float = 1.0):
    if not h_hats:
        return torch.tensor(0.0, device=h.device)

    B, T, d = h.shape
    d_steps = len(h_hats)
    losses = []

    for step_idx, h_hat in enumerate(h_hats):
        i = step_idx + 1
        target = h[:, i:T - d_steps + i].detach()
        mask_slice = mask[:, i:T - d_steps + i]
        loss_elem = F.smooth_l1_loss(h_hat, target, reduction='none', beta=beta)
        loss_feat_avg = loss_elem.mean(dim=-1)
        masked_loss = loss_feat_avg[mask_slice]
        if masked_loss.numel() > 0:
            losses.append(masked_loss.mean())

    if not losses:
        return torch.tensor(0.0, device=h.device)

    return torch.stack(losses).mean()

# ── Step 018  kl_alignment_loss ──
def kl_alignment_loss(h, h_hats: list, mask, params: dict):
    if not h_hats:
        return torch.tensor(0.0, device=h.device)

    B, T, d = h.shape
    d_steps = len(h_hats)
    losses = []

    head_w = params['head_w'].detach()
    head_b = params['head_b'].detach()

    for step_idx, h_hat in enumerate(h_hats):
        i = step_idx + 1
        h_true = h[:, i:T - d_steps +i].detach()
        logits_true = h_true @ head_w + head_b
        logits_pred = h_hat @ head_w + head_b
        log_probs_true = F.log_softmax(logits_true, dim=-1)
        log_probs_pred = F.log_softmax(logits_pred, dim=-1)
        kl_elem = (log_probs_true.exp() * (log_probs_true - log_probs_pred)).sum(dim=-1)
        mask_slice = mask[:, i:T - d_steps + i]
        masked_kl = kl_elem[mask_slice]
        if masked_kl.numel() > 0:
            losses.append(masked_kl.mean())

    if not losses:
        return torch.tensor(0.0, device=h.device)

    return torch.stack(losses).mean()

# ── Step 019  nextlat_loss ──
def nextlat_loss(batch: dict, params: dict, dyn: dict, n_heads: int, d_steps: int,
                 lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    x = batch['x']
    y = batch['y']
    mask = batch['mask']

    h = gpt_hidden_states(x, params, n_heads)
    logits = output_head(h, params)
    next_token = next_token_loss(logits, y, mask)

    eos = params['head_b'].shape[0] - 1
    mask_x = x != eos

    if d_steps > 0:
        h_hats = rollout_latents(h, x, params, dyn, d_steps)
        next_h = next_hidden_loss(h, h_hats, mask_x, beta=beta)
        kl = kl_alignment_loss(h, h_hats, mask_x, params)
    else:
        next_h = torch.tensor(0.0, device=h.device)
        kl = torch.tensor(0.0, device=h.device)

    total = next_token + lam_h * next_h + lam_kl * kl

    return {
        'total': total,
        'next_token': next_token,
        'next_h': next_h,
        'kl': kl,
    }

# ── Step 020  train_step ──
def train_step(batch: dict, params: dict, dyn: dict, opt, n_heads: int, d_steps: int,
               lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    opt.zero_grad()

    losses = nextlat_loss(batch, params, dyn, n_heads, d_steps, lam_h, lam_kl, beta)

    losses['total'].backward()

    opt.step()

    return {
        'total': float(losses['total'].item()),
        'next_token': float(losses['next_token'].item()),
        'next_h': float(losses['next_h'].item()),
        'kl': float(losses['kl'].item())
    }

# ── Step 021  train_model ──
def train_model(dataset: dict, cfg: dict, seed: int = 0) -> tuple:
    G = dataset['G']
    T = dataset['tokens'].shape[1]
    vocab_size = 4 + G * G + 1

    params = init_gpt_params(vocab_size, cfg['d_model'], cfg['n_layers'], T, seed)
    dyn = init_dynamics_params(cfg['d_model'], cfg['hidden'], seed)

    all_params = list(params.values()) + list(dyn.values())
    opt = torch.optim.Adam(all_params, lr=cfg['lr'])

    history = []

    for step in range(cfg['steps']):
        batch = get_batch(dataset, cfg['batch_size'], step)
        losses = train_step(
            batch, params, dyn, opt,
            n_heads=cfg['n_heads'],
            d_steps=cfg['d_steps'],
            lam_h=cfg['lam_h'],
            lam_kl=cfg['lam_kl'],
            beta=cfg['beta'],
        )
        history.append(losses)

    return params, dyn, history

# ── Step 022  greedy_decode ──
def greedy_decode(params: dict, n_heads: int, prefix: list, n_tokens: int) -> list:
    tokens = torch.tensor([prefix], dtype=torch.long)
    generated = []

    with torch.no_grad():
        for _ in range(n_tokens):
            h = gpt_hidden_states(tokens, params, n_heads)
            logits = output_head(h[:, -1:, :], params)
            next_token = torch.argmin(-logits.squeeze(0), dim=-1)
            next_token = int(next_token.item())
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    return generated

# ── Step 023  effective_rank ──
import torch
import math

def effective_rank(H, tol: float = 1e-12) -> float:
    sv = torch.linalg.svdvals(H)
    sv = sv[sv > tol]

    if len(sv) == 0:
        return 0.0

    sv_norm = sv / sv.sum()

    entropy = -torch.sum(sv_norm * torch.log(sv_norm))

    return float(torch.exp(entropy))

# ── Step 024  eval_hidden_states ──
import torch

def eval_hidden_states(dataset: dict, params: dict, n_heads: int, n_rows: int):
    tokens = dataset['tokens'][:n_rows]
    mask = dataset['mask'][:n_rows]

    x = tokens[:, :-1]
    mask_x = mask[:, :-1]

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)

    N = int(mask_x.sum().item())
    h_flat = h.reshape(-1, h.shape[-1])
    mask_flat = mask_x.reshape(-1)
    return h_flat[mask_flat]

# ── Step 025  valid_move_rate ──
import torch

def valid_move_rate(dataset: dict, params: dict, n_heads: int, n_rows: int) -> float:
    tokens = dataset['tokens'][:n_rows]
    mask = dataset['mask'][:n_rows]
    states = dataset['states'][:n_rows]
    G = dataset['G']
    EOS = 4 + G * G

    x = tokens[:, :-1]
    y_mask = mask[:, 1:]
    states_before = states[:, :-1]

    goal_idx = tokens[:, 1] - 4
    goal_idx_expanded = goal_idx.unsqueeze(1).expand(-1, states_before.shape[1])

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        logits = output_head(h, params)
        preds = torch.argmax(logits, dim=-1)

    valid = 0
    total = 0
    T_minus_1 = x.shape[1]
    for i in range(x.shape[0]):
        for t in range(1, T_minus_1):
            if not y_mask[i, t]:
                continue
            total += 1
            pos_cell = int(states_before[i, t].item())
            goal_cell = int(goal_idx_expanded[i, t].item())
            pred = int(preds[i, t].item())

            if pos_cell == goal_cell:
                if pred == EOS:
                    valid += 1
            else:
                row = pos_cell // G
                col = pos_cell % G
                if pred in legal_actions((row, col), G):
                    valid += 1

    if total == 0:
        return 0.0
    return valid / total

# ── Step 026  sequence_compression ──
import torch

def sequence_compression(dataset: dict, params: dict, n_heads: int, n_tokens: int, max_pairs: int) -> float:
    tokens = dataset['tokens']
    mask = dataset['mask']
    states = dataset['states']
    G = dataset['G']
    n_rows = tokens.shape[0]
    T = tokens.shape[1]
    
    groups = {}
    
    for i in range(n_rows):
        goal = int(tokens[i, 1].item()) - 4
        for t in range(2, T):
            if not mask[i, t].item():
                continue
            if tokens[i, t].item() >= 4:
                continue
            if t + 1 + n_tokens > T:
                continue
            
            state = int(states[i, t].item())
            key = (state, goal)
            prefix = tokens[i, :t+1].tolist()
            
            if key not in groups:
                groups[key] = []
            
            if len(groups[key]) < 2:
                if len(groups[key]) == 0:
                    groups[key].append(prefix)
                else:
                    if groups[key][0] != prefix:
                        groups[key].append(prefix)
    
    pair_keys = [key for key, prefixes in groups.items() if len(prefixes) == 2]
    pair_keys = pair_keys[:max_pairs]
    
    if not pair_keys:
        return 0.0
    
    matches = 0
    for key in pair_keys:
        prefixes = groups[key]
        cont1 = greedy_decode(params, n_heads, prefixes[0], n_tokens)
        cont2 = greedy_decode(params, n_heads, prefixes[1], n_tokens)
        
        if cont1 == cont2:
            matches += 1
    
    return matches / len(pair_keys)

# ── Step 027  detour_robustness ──
import numpy as np
import torch

def detour_robustness(params: dict, n_heads: int, G: int, max_steps: int, n_trials: int,
                      detour_prob: float = 0.75, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    successes = 0
    EOS = 4 + G * G
    
    for _ in range(n_trials):
        start_row = rng.integers(0, G)
        start_col = rng.integers(0, G)
        goal_row = rng.integers(0, G)
        goal_col = rng.integers(0, G)
        
        start = (start_row, start_col)
        goal = (goal_row, goal_col)
        
        start_cell = 4 + start_row * G + start_col
        goal_cell = 4 + goal_row * G + goal_col
        seq = [start_cell, goal_cell]
        pos = start
        reached_goal = False
        
        for _ in range(max_steps):
            if pos == goal:
                reached_goal = True
                break
            
            if rng.random() < detour_prob:
                legal = legal_actions(pos, G)
                action = int(rng.choice(legal))
            else:
                preds = greedy_decode(params, n_heads, seq, 1)
                action = preds[0]
                
                if action < 0 or action > 3:
                    reached_goal = False
                    break
                
                legal = legal_actions(pos, G)
                if action not in legal:
                    reached_goal = False
                    break
            
            pos, legal = grid_step(pos, action, G)
            if not legal:
                reached_goal = False
                break
            
            seq.append(action)
        
        if pos == goal:
            successes += 1
    
    return successes / n_trials

# ── Step 028  world_model_report ──
def world_model_report(dataset: dict, params: dict, n_heads: int, n_rows: int, n_tokens: int,
                       max_pairs: int, n_trials: int, seed: int = 0) -> dict:
    G = dataset['G']
    T = dataset['tokens'].shape[1]

    vmr = valid_move_rate(dataset, params, n_heads, n_rows)

    H = eval_hidden_states(dataset, params, n_heads, n_rows)
    er = effective_rank(H)

    sc = sequence_compression(dataset, params, n_heads, n_tokens, max_pairs)

    dr = detour_robustness(params, n_heads, G, max_steps=T - 3, n_trials=n_trials, detour_prob=0.75, seed=seed)

    return {
        'valid_move_rate': round(float(vmr), 4),
        'effective_rank': round(float(er), 4),
        'sequence_compression': round(float(sc), 4),
        'detour_robustness': round(float(dr), 4),
    }

# ── Step 029  draft_from_latent ──
def draft_from_latent(h_last, dyn: dict, params: dict, max_draft: int) -> tuple:
    with torch.no_grad():
        logits = output_head(h_last.unsqueeze(0), params)
        next_token = int(torch.argmax(logits, dim=-1).item())

        drafts = []
        h = h_last
        cur = next_token
        wte = params['wte']

        for _ in range(max_draft):
            emb = wte[cur]
            h = latent_transition(h.unsqueeze(0), emb.unsqueeze(0), dyn).squeeze(0)
            logits = output_head(h.unsqueeze(0), params)
            cur = int(torch.argmax(logits, dim=-1).item())
            drafts.append(cur)

    return next_token, drafts

# ── Step 030  verify_draft ──
import torch

def verify_draft(params: dict, n_heads: int, prefix: list, next_token: int, drafts: list) -> tuple:
    seq = prefix + [next_token] + drafts
    tokens = torch.tensor([seq], dtype=torch.long)

    with torch.no_grad():
        h = gpt_hidden_states(tokens, params, n_heads)
        logits = output_head(h, params)
        preds = torch.argmax(logits, dim=-1).squeeze(0)

    base = len(prefix)
    n_accepted = 0
    for j, draft in enumerate(drafts):
        if int(preds[base + j].item()) == draft:
            n_accepted += 1
        else:
            break

    corr = int(preds[base + n_accepted].item())
    return n_accepted, corr

# ── Step 031  self_speculative_generate ──
def self_speculative_generate(params: dict, dyn: dict, n_heads: int, prefix: list,
                              n_tokens: int, max_draft: int) -> dict:
    seq = list(prefix)
    start_len = len(seq)
    max_len = params['wpe'].shape[0]

    cycles = 0
    accepted = []

    while len(seq) - start_len < n_tokens:
        with torch.no_grad():
            h_last = gpt_hidden_states(
                torch.tensor([seq], dtype=torch.long), params, n_heads
            )[0, -1]

        k = max(0, min(max_draft, max_len - len(seq) - 2))
        next_token, drafts = draft_from_latent(h_last, dyn, params, k)
        n_accepted, correction = verify_draft(params, n_heads, seq, next_token, drafts)

        seq.extend([next_token] + drafts[:n_accepted] + [correction])
        accepted.append(n_accepted)
        cycles += 1

    return {
        'tokens': seq[start_len:start_len + n_tokens],
        'cycles': cycles,
        'accepted': accepted,
    }

# ── Step 032  speculative_stats ──
def speculative_stats(result: dict, n_tokens: int) -> dict:
    cycles = result['cycles']
    accepted_list = result['accepted']

    if cycles == 0:
        mean_accepted = 0.0
        speedup = 0.0
    else:
        mean_accepted = sum(accepted_list) / cycles
        speedup = n_tokens / cycles

    return {
        'cycles': cycles,
        'mean_accepted': round(float(mean_accepted), 4),
        'speedup': round(float(speedup), 4)
    }

# ── Scaffold (runner) ──
"""End-to-end NextLat experiment (Teoh et al., arXiv:2511.05963) on a grid world.

Story: train the SAME tiny transformer twice on goal-directed random walks -
once with plain next-token prediction (GPT), once with NextLat's auxiliary
next-latent objective - then score both the way the paper does: next-token
legality, effective latent rank, sequence compression and detour robustness.
Finally, use NextLat's latent dynamics model as a free draft model for
variable-length self-speculative decoding and measure accepted tokens and
speedup. Greedy verification makes the speculative output identical to plain
greedy decoding, which the script checks.
"""
import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    G, T = 4, 14
    n_heads = 4
    train_ds = make_dataset(n=1024, G=G, T=T, seed=0)
    eval_ds = make_dataset(n=256, G=G, T=T, seed=1)
    print(f"grid {G}x{G}, T={T}, vocab={4 + G * G + 1}, train sequences={train_ds['tokens'].shape[0]}")

    base = dict(d_model=32, n_layers=2, n_heads=n_heads, hidden=64, steps=250, batch_size=32, lr=3e-3, beta=1.0)
    cfg_gpt = dict(base, d_steps=0, lam_h=0.0, lam_kl=0.0)
    cfg_nextlat = dict(base, d_steps=2, lam_h=1.0, lam_kl=0.5)

    # ---- 1. Train both objectives from the same initialization ----
    p_gpt, _dyn_unused, hist_gpt = train_model(train_ds, cfg_gpt, seed=0)
    print(f"GPT      next-token loss: {hist_gpt[0]['next_token']:.3f} -> {hist_gpt[-1]['next_token']:.3f}")
    p_nl, dyn_nl, hist_nl = train_model(train_ds, cfg_nextlat, seed=0)
    print(f"NextLat  next-token loss: {hist_nl[0]['next_token']:.3f} -> {hist_nl[-1]['next_token']:.3f}"
          f" | next-hidden {hist_nl[0]['next_h']:.4f} -> {hist_nl[-1]['next_h']:.4f}"
          f" | KL {hist_nl[0]['kl']:.4f} -> {hist_nl[-1]['kl']:.4f}")

    # ---- 2. World-model metrics (the paper's Table 1) ----
    kw = dict(n_heads=n_heads, n_rows=128, n_tokens=4, max_pairs=60, n_trials=40, seed=2)
    rep_gpt = world_model_report(eval_ds, p_gpt, **kw)
    rep_nl = world_model_report(eval_ds, p_nl, **kw)
    print("\nmetric                   GPT      NextLat")
    for key, arrow in [("valid_move_rate", "up"), ("effective_rank", "down"),
                       ("sequence_compression", "up"), ("detour_robustness", "up")]:
        print(f"{key:22s} {rep_gpt[key]:8.4f} {rep_nl[key]:8.4f}   (better = {arrow})")
    print("(lower effective rank = more compact latent state; the true world has only "
          f"{G * G} positions x {G * G} goals)")

    # ---- 3. Variable-length self-speculative decoding with NextLat's dynamics ----
    n_tokens, max_draft = 6, 4
    accepted, speedups, lossless = [], [], True
    for i in range(8):
        prefix = eval_ds["tokens"][i, :2].tolist()  # [start_cell, goal_cell]
        res = self_speculative_generate(p_nl, dyn_nl, n_heads, prefix, n_tokens, max_draft)
        stats = speculative_stats(res, n_tokens)
        accepted.append(stats["mean_accepted"])
        speedups.append(stats["speedup"])
        lossless = lossless and (res["tokens"] == greedy_decode(p_nl, n_heads, prefix, n_tokens))
    print(f"\nself-speculative decoding ({n_tokens} tokens, draft up to {max_draft}):")
    print(f"  mean accepted drafts per cycle: {float(np.mean(accepted)):.2f}")
    print(f"  speedup in transformer passes:  {float(np.mean(speedups)):.2f}x")
    print(f"  identical to greedy decoding:   {lossless}")
    print("\nnote: at this toy scale two effects are robust across seeds - NextLat's lower effective "
          "rank (a more compact latent state) and its usable drafts (GPT's untrained dynamics accept "
          "almost none). The other metrics are noisy on 60 pairs / 40 episodes; rerun with more "
          "training steps, sequences and trials before reading anything into small gaps.")


if __name__ == "__main__":
    main()
