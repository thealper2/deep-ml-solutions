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