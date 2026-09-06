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