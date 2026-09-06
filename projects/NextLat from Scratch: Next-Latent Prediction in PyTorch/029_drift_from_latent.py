def draft_from_latent(h_last, dyn: dict, params: dict, max_draft: int) -> tuple:
    with torch.no_grad():
        logits = output_head(h_last.unsqueeze(0), params)
        next_token = int(torch.argmax(logits, dim=-1).item())

        drafts = []
        h = h_last
        wte = params['wte']

        for _ in range(max_draft):
            emb = wte[next_token]
            h = latent_transition(h.unsqueeze(0), emb.unsqueeze(0), dyn)
            h = h.squeeze(0)

            logits = output_head(h.unsqueeze(0), params)
            next_token = int(torch.argmax(logits, dim=-1).item())
            drafts.append(next_token)

    return next_token, drafts