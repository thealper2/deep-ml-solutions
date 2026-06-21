def build_id_to_token_vocab(token_to_id):
    return {v: k for k, v in token_to_id.items()}
