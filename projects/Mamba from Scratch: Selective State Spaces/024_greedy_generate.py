def greedy_generate(prompt_ids, params, max_new_tokens):
    """Greedily generate new token ids from a prompt using a carried SSM cache."""
    prompt_ids = prompt_ids.clone().detach().long()
    generated = [prompt_ids]

    cache = None

    for i in range(len(prompt_ids)):
        logits, cache = mamba_recurrent_step(prompt_ids[i:i+1], params, cache)

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1)
        generated.append(next_token)
        logits, cache = mamba_recurrent_step(next_token, params, cache)

    result = torch.cat(generated, dim=0)
    return result