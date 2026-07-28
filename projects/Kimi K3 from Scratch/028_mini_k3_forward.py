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
