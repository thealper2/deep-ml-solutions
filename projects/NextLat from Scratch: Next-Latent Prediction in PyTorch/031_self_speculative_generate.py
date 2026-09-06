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