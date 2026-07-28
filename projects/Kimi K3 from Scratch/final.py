"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  short_conv ──
def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    T, d = x.shape
    K = w.shape[0]
    y = np.zeros_like(x)

    for j in range(K):
        shift = K - 1 - j
        y[shift:] += w[j] * x[:-shift] if shift > 0 else w[j] * x

    return y

# ── Step 002  kda_qkv ──
def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    q_proj = x @ params['Wq']
    q_conv = short_conv(q_proj, params['cq'])
    q_swish = q_conv * (1 / (1 + np.exp(-q_conv)))
    q = q_swish / np.sqrt(np.sum(q_swish ** 2, axis=1, keepdims=True) + 1e-6)

    k_proj = x @ params['Wk']
    k_conv = short_conv(k_proj, params['ck'])
    k_swish = k_conv * (1 / (1 + np.exp(-k_conv)))
    k = k_swish / np.sqrt(np.sum(k_swish ** 2, axis=1, keepdims=True) + 1e-6)

    v_proj = x @ params['Wv']
    v_conv = short_conv(v_proj, params['cv'])
    v = v_conv * (1 / (1 + np.exp(-v_conv)))
    return q, k, v

# ── Step 003  kda_gates ──
def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    beta = 1 / (1 + np.exp(-(x @ params['wb'] + params['bb'])))
    z = x @ params['Wd1'] @ params['Wd2'] + params['ba']
    return beta, z

# ── Step 004  lower_bounded_decay ──
def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    log_scale = np.exp(A)
    g = g_min * (1 / (1 + np.exp(-log_scale * z)))
    return np.exp(g)

# ── Step 005  kda_state_update ──
def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    Sd = alpha[:, None] * S
    readout = k @ Sd
    erase = beta * np.outer(k, readout)
    write = beta * np.outer(k, v)
    return Sd - erase + write

# ── Step 006  kda_recurrence ──
def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    T, dk = q.shape
    dv = v.shape[1]

    if S0 is None:
        S = np.zeros((dk, dv))
    else:
        S = S0.copy()

    O = np.zeros((T, dv))

    for t in range(T):
        S = kda_state_update(S, k[t], v[t], alpha[t], beta[t])
        O[t] = S.T @ q[t]

    return O, S

# ── Step 007  cumulative_decay ──
def cumulative_decay(alpha):
    """Inclusive channel-wise cumulative product of alpha down the time axis.

    alpha: (C, dk) per-step retention factors -> Gamma: (C, dk).
    """
    return np.cumprod(alpha, axis=0)

# ── Step 008  chunk_pseudo_values ──
def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    C, dk = k.shape
    dv = v.shape[1]

    Gamma = cumulative_decay(alpha)
    Khat = k * Gamma
    Kcheck = k / Gamma

    M = np.tril(Khat @ Kcheck.T, k=-1)

    rhs = beta[:, None] * (v - Khat @ S0)

    L = np.eye(C) + np.diag(beta) @ M
    U = np.linalg.solve(L, rhs)

    return U

# ── Step 009  kda_chunkwise ──
def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. Returns (O, S_final).
    """
    T, dk = q.shape
    dv = v.shape[1]
    
    if S0 is None:
        S = np.zeros((dk, dv))
    else:
        S = S0.copy()
    
    O = np.zeros((T, dv))
    
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        C = end - start
        
        q_c = q[start:end]
        k_c = k[start:end]
        v_c = v[start:end]
        alpha_c = alpha[start:end]
        beta_c = beta[start:end]
        
        Gamma = cumulative_decay(alpha_c)
        Qhat = q_c * Gamma
        Kcheck = k_c / Gamma
        
        U = chunk_pseudo_values(k_c, v_c, alpha_c, beta_c, S)
        
        inter = Qhat @ S
        
        M = Qhat @ Kcheck.T
        tril_M = np.tril(M)
        intra = tril_M @ U
        
        O[start:end] = inter + intra
        
        S = Gamma[-1][:, None] * (S + Kcheck.T @ U)
    
    return O, S

# ── Step 010  kda_output_gate ──
def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    rms = np.sqrt(np.mean(o ** 2, axis=1, keepdims=True) + 1e-6)
    o_norm = o / rms
    gate = 1 / (1 + np.exp(-(x @ Wg)))
    return (gate * o_norm) @ Wo

# ── Step 011  mla_compress_reconstruct ──
def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    c = x @ Wc
    K = (c @ Wk_up).reshape(x.shape[0], n_heads, -1)
    V = (c @ Wv_up).reshape(x.shape[0], n_heads, -1)
    return c, K, V

# ── Step 012  nope_attention ──
def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    T, d = x.shape
    dh = Wq.shape[1] // n_heads
    Q = (x @ Wq).reshape(T, n_heads, dh)
    _, K, V = mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads)
    out = np.zeros((T, n_heads, dh))

    for h in range(n_heads):
        Q_h = Q[:, h, :]
        K_h = K[:, h, :]
        V_h = V[:, h, :]

        scores = (Q_h @ K_h.T) / np.sqrt(dh)

        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores[mask] = -1e9

        scores_stable = scores - np.max(scores, axis=1, keepdims=True)
        attn = np.exp(scores_stable)
        attn = attn / np.sum(attn, axis=1, keepdims=True)

        out[:, h, :] = attn @ V_h

    return out.reshape(T, -1)

# ── Step 013  mla_output_gate ──
def mla_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * o) @ Wo - note: no RMSNorm here, unlike KDA's gate.

    o: (T, H*dh) attention output.  x: (T, d) layer input.  Returns (T, d).
    """
    gate = 1 / (1 + np.exp(-(x @ Wg)))
    return (gate * o) @ Wo

# ── Step 014  hybrid_schedule ──
def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    pattern = ['KDA', 'KDA', 'KDA', 'MLA']
    return pattern * n_repeats + ['MLA']

# ── Step 015  attnres_weights ──
def attnres_weights(pseudo_q, sources):
    """Softmax over depth: w[i, t] prop. to exp(pseudo_q . RMSNorm(sources[i][t])).

    sources: list of n (T, d) arrays.  Returns (n, T); columns sum to 1.
    """
    n = len(sources)
    T = sources[0].shape[0]

    logits = np.zeros((n, T))
    for i, src in enumerate(sources):
        rms = np.sqrt(np.mean(src ** 2, axis=1, keepdims=True) + 1e-6)
        src_norm = src / rms
        logits[i] = src_norm @ pseudo_q

    max_logits = np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    weights = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    return weights

# ── Step 016  attnres_full ──
def attnres_full(pseudo_q, sources):
    """h[t] = sum_i attnres_weights(...)[i, t] * sources[i][t] (raw values).

    Returns (T, d).
    """
    weights = attnres_weights(pseudo_q, sources)
    n, T = weights.shape
    d = sources[0].shape[1]

    h = np.zeros((T, d))
    for i in range(n):
        h += weights[i, :, None] * sources[i]

    return h

# ── Step 017  block_partial_sums ──
def block_partial_sums(layer_outputs):
    """Running sums of a block's layer outputs; entry i sums outputs 0..i.

    Returns a list of independent (T, d) arrays; last entry = block sum b_n.
    """
    running = []
    cumsum = np.zeros_like(layer_outputs[0])
    for out in layer_outputs:
        cumsum = cumsum + out
        running.append(cumsum.copy())

    return running

# ── Step 018  attnres_block ──
def attnres_block(pseudo_q, block_reps, partial):
    """Full AttnRes over [b_0..b_{n-1}] plus the current block's partial (if any).

    partial is None for the first layer of a block. Returns (T, d).
    """
    sources = list(block_reps)
    if partial is not None:
        sources.append(partial)
        
    return attnres_full(pseudo_q, sources)

# ── Step 019  situ_glu ──
def situ_glu(x, Wg, Wu, beta1=4.0, beta2=25.0):
    """(softcap(x@Wg, b1) * sigmoid(x@Wg)) * softcap(x@Wu, b2), softcap = b*tanh(u/b).

    Use sigmoid(g) = 0.5*(1 + tanh(0.5*g)) to stay overflow-safe. |out| <= b1*b2.
    """
    g = x @ Wg
    u = x @ Wu

    softcap_g = beta1 * np.tanh(g / beta1)
    softcap_u = beta2 * np.tanh(u / beta2)

    sigmoid_g = 0.5 * (1 + np.tanh(0.5 * g))
    
    return softcap_g * sigmoid_g * softcap_u

# ── Step 020  route_topk ──
def route_topk(x, Wr, bias, k):
    """s = sigmoid(x @ Wr); top-k by s + bias (stable, descending);
    p = raw selected scores normalized per token. Returns (s, idx, p).
    """
    s = 1 / (1 + np.exp(-(x @ Wr)))
    biased = s + bias
    idx = np.argsort(-biased, axis=1, kind='stable')[:, :k]
    p_raw = np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        p_raw[i] = s[i, idx[i]]

    p = p_raw / np.sum(p_raw, axis=1, keepdims=True)
    return s, idx, p

# ── Step 021  routed_experts ──
def routed_experts(z, idx, p, experts):
    """u[t] = sum_j p[t,j] * situ_glu(z[t], *experts[idx[t,j]]).

    z: (T, l).  experts: list of (Wg, Wu) latent-width SiTU-GLUs. Returns (T, l).
    """
    T, l = z.shape
    k = idx.shape[1]
    u = np.zeros((T, l))

    for t in range(T):
        for j in range(k):
            expert_idx = idx[t, j]
            Wg, Wu = experts[expert_idx]
            expert_out = situ_glu(z[t:t+1], Wg, Wu)
            u[t] += p[t, j] * expert_out[0]

    return u

# ── Step 022  stable_latent_moe ──
def stable_latent_moe(x, params):
    """y = sum_shared SiTU(x) + RMSNorm(routed_aggregate) @ Wup (Eq. 11).

    Route on full-width x; compute in latent width; RMSNorm(u) before Wup.
    """
    d = x.shape[1]
    l = params['Wdown'].shape[1]
    z = x @ params['Wdown']
    s, idx, p = route_topk(x, params['Wr'], params['bias'], params['k'])
    u = routed_experts(z, idx, p, params['experts'])
    rms = np.sqrt(np.mean(u**2, axis=1, keepdims=True) + 1e-6)
    u_norm = u / rms
    shared_sum = np.zeros((x.shape[0], d))
    for Wg, Wu in params['shared']:
        shared_sum += situ_glu(x, Wg, Wu)
    
    y = shared_sum + u_norm @ params['Wup']
    return y

# ── Step 023  topk_cutoffs ──
def topk_cutoffs(s, bias, k):
    """Top-(k+1) on s + bias: first k -> routes (m, k); (k+1)-th biased score
    -> cutoffs (m,). Returns (routes, cutoffs).
    """
    biased = s + bias
    idx_sorted = np.argsort(-biased, axis=1, kind='stable')
    routes = idx_sorted[:, :k]
    cutoffs = biased[np.arange(biased.shape[0]), idx_sorted[:, k]]
    return routes, cutoffs

# ── Step 024  quantile_balance_update ──
def quantile_balance_update(s, bias, k):
    """QB (Eq. 14): bhat_j = -(the (q+1)-th largest of s[:, j] - cutoffs),
    q = m*k // n; return bhat - mean(bhat).
    """
    m, n = s.shape
    q = (m * k) // n
    routes, cutoffs = topk_cutoffs(s, bias, k)
    margins = s - cutoffs[:, None]
    bhat = np.zeros(n)
    for j in range(n):
        col_margins = margins[:, j]
        sorted_margins = np.sort(col_margins)[::-1]
        bhat[j] = -sorted_margins[q]

    bhat = bhat - np.mean(bhat)
    return bhat

# ── Step 025  histogram_quantile ──
def histogram_quantile(x, n_bins, lo, hi, q_frac):
    """Quantile from pooled bin counts; error <= (hi - lo) / n_bins.

    Return the right edge of the first bin whose cumulative count reaches
    q_frac * len(x).
    """
    counts, bin_edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    cumsum = np.cumsum(counts)
    target = q_frac * len(x)
    idx = np.searchsorted(cumsum, target)
    if idx >= n_bins:
        return float(bin_edges[-1])

    return float(bin_edges[idx + 1])

# ── Step 026  newton_schulz ──
def newton_schulz(G, n_iters=5):
    """Muon's Newton-Schulz orthogonalization (a,b,c = 3.4445, -4.7750, 2.0315).

    Normalize by the Frobenius norm (+1e-7), iterate the quintic, transpose
    handling for tall matrices. Singular values -> 1.
    """
    transposed = False
    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True

    frob = np.sqrt(np.sum(G ** 2))
    X = G / (frob + 1e-7)

    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X

# ── Step 027  per_head_muon ──
def per_head_muon(M, n_heads, n_iters=5):
    """Split M (d, H*dh) into H column blocks, newton_schulz each, re-concatenate."""
    d, total = M.shape
    dh = total // n_heads
    
    blocks = []
    for h in range(n_heads):
        start = h * dh
        end = (h + 1) * dh
        block = M[:, start:end]
        block_orth = newton_schulz(block, n_iters)
        blocks.append(block_orth)
    
    return np.hstack(blocks)

# ── Step 028  mini_k3_forward ──
def mini_k3_forward(tokens, seed):
    """Miniature Kimi K3 forward pass. Returns (T, 17) logits.

    Parameter creation order (all rng.normal(0.0, 0.3, size=...)):
      emb (17, 16)
      then for each layer of hybrid_schedule(1):
        wq_res (16,)
        KDA: Wq (16,8), Wk (16,8), Wv (16,8), cq (3,8), ck (3,8), cv (3,8),
             wb (16,), Wd1 (16,4), Wd2 (4,8), ba (8,), Wg (16,8), Wo (8,16)
             (bb = 0.0 and A = 0.0 are constants, not drawn)
        MLA: Wq (16,16), Wc (16,8), Wk_up (8,16), Wv_up (8,16),
             Wg (16,16), Wo (16,16)
        MoE: Wdown (16,8), Wup (8,16), Wr (16,8), bias = zeros(8), k=2,
             experts: 8 x (Wg (8,8), Wu (8,8)) drawn gate-then-up per expert,
             shared: 2 x (Wg (16,16), Wu (16,16)) drawn gate-then-up
      w_final (16,)
    """
    rng = np.random.default_rng(seed)
    def W(*shape):
        return rng.normal(0.0, 0.3, size=shape)

    emb = W(17, 16)
    sources = [emb[tokens]]

    for layer_type in hybrid_schedule(1):
        wq_res = W(16)

        if layer_type == 'KDA':
            Wq = W(16, 8); Wk = W(16, 8); Wv = W(16, 8)
            cq = W(3, 8); ck = W(3, 8); cv = W(3, 8)
            wb = W(16); Wd1 = W(16, 4); Wd2 = W(4, 8); ba = W(8)
            Wg = W(16, 8); Wo = W(8, 16)
            bb = 0.0; A = 0.0
        else:
            Wq = W(16, 16); Wc = W(16, 8); Wk_up = W(8, 16); Wv_up = W(8, 16)
            Wg = W(16, 16); Wo = W(16, 16)

        Wdown = W(16, 8); Wup = W(8, 16); Wr = W(16, 8); bias = np.zeros(8)
        experts = [(W(8, 8), W(8, 8)) for _ in range(8)]
        shared = [(W(16, 16), W(16, 16)) for _ in range(2)]
        moe_params = {'Wdown': Wdown, 'Wup': Wup, 'Wr': Wr, 'bias': bias,
                      'k': 2, 'experts': experts, 'shared': shared}

        h = attnres_full(wq_res, sources)

        if layer_type == 'KDA':
            q, k, v = kda_qkv(h, {'Wq': Wq, 'Wk': Wk, 'Wv': Wv,
                                  'cq': cq, 'ck': ck, 'cv': cv})
            beta, z = kda_gates(h, {'wb': wb, 'bb': bb, 'Wd1': Wd1, 'Wd2': Wd2, 'ba': ba})
            alpha = lower_bounded_decay(z, A)
            O, _ = kda_recurrence(q, k, v, alpha, beta)
            a = kda_output_gate(O, h, Wg, Wo)
        else:
            o = nope_attention(h, Wq, Wc, Wk_up, Wv_up, n_heads=2)
            a = mla_output_gate(o, h, Wg, Wo)

        f = a + stable_latent_moe(a, moe_params)
        sources.append(f)

    w_final = W(16)
    h_out = attnres_full(w_final, sources)
    return h_out @ emb.T

# ── Scaffold (runner) ──
"""Mini Kimi K3: KDA + Gated MLA + Attention Residuals + Stable LatentMoE."""

import numpy as np


def main():
    rng = np.random.default_rng(0)

    # --- KDA: chunkwise form must equal the recurrence ---
    T, dk, dv = 12, 4, 3
    q = rng.normal(size=(T, dk))
    k = rng.normal(size=(T, dk))
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    v = rng.normal(size=(T, dv))
    alpha = lower_bounded_decay(rng.normal(size=(T, dk)), 0.0)
    beta = 1.0 / (1.0 + np.exp(-rng.normal(size=T)))
    O_rec, S_rec = kda_recurrence(q, k, v, alpha, beta)
    O_ch, S_ch = kda_chunkwise(q, k, v, alpha, beta, chunk_size=4)
    print("chunkwise == recurrence:", bool(np.allclose(O_ch, O_rec, atol=1e-9)))
    print("alpha bounds ok:", bool((alpha >= np.exp(-5.0) - 1e-12).all() and (alpha < 1.0).all()))

    # --- Attention Residuals: full vs block source counts ---
    srcs = [rng.normal(size=(5, 8)) for _ in range(6)]
    pq = rng.normal(size=8)
    h_full = attnres_full(pq, srcs)
    partials = block_partial_sums(srcs[3:])
    h_block = attnres_block(pq, [srcs[0], srcs[1] + srcs[2]], partials[-1])
    print("attnres full shape:", h_full.shape, "block shape:", h_block.shape)

    # --- SiTU-GLU boundedness ---
    big = rng.normal(size=(10, 6)) * 1000
    out = situ_glu(big, rng.normal(size=(6, 5)), rng.normal(size=(6, 5)))
    print("SiTU-GLU max |out| (bound 100):", round(float(np.abs(out).max()), 2))

    # --- Quantile Balancing on the demo batch ---
    s = 1.0 / (1.0 + np.exp(-np.random.default_rng(12).normal(size=(16, 4)) * 2))
    before = np.bincount(np.argsort(-s, axis=1)[:, :1].ravel(), minlength=4)
    b1 = quantile_balance_update(s, np.zeros(4), 1)
    after = np.bincount(np.argsort(-(s + b1), axis=1)[:, :1].ravel(), minlength=4)
    print("QB loads before:", before.tolist(), "after:", after.tolist())
    margins = rng.normal(size=4000)
    est = histogram_quantile(margins, 200, -5.0, 5.0, 0.75)
    print("histogram quantile err <= binwidth:",
          bool(abs(est - np.quantile(margins, 0.75)) <= 10.0 / 200 + 1e-12))

    # --- Per-head Muon ---
    G = rng.normal(size=(16, 8))
    M = G.copy(); M[:, :4] *= 100.0
    ph = per_head_muon(M, 2, n_iters=3)
    fu = newton_schulz(M, n_iters=3)
    print("per-head norm ratio:",
          round(float(np.linalg.norm(ph[:, 4:]) / np.linalg.norm(ph[:, :4])), 3),
          "| full-matrix:",
          round(float(np.linalg.norm(fu[:, 4:]) / np.linalg.norm(fu[:, :4])), 3))

    # --- The mini model, end to end ---
    print("schedule:", hybrid_schedule(1))
    tokens = [1, 5, 3, 7, 2, 0]
    logits = mini_k3_forward(tokens, seed=0)
    print("logits shape:", logits.shape, "finite:", bool(np.isfinite(logits).all()))
    edited = mini_k3_forward([1, 5, 3, 7, 2, 9], seed=0)
    print("causal end-to-end:", bool(np.allclose(logits[:5], edited[:5])))
    print("next-token argmax per position:", np.argmax(logits, axis=1).tolist())


if __name__ == "__main__":
    main()
