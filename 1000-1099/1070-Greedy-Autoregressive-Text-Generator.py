import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    generated_ids = list(idx)

    for _ in range(max_new_tokens):
        context = generated_ids[-context_size:] if len(generated_ids) > context_size else generated_ids
        context = np.array(context).reshape(1, -1)
        next_token_probs = model(context)
        next_token_id = int(np.argmax(next_token_probs[0, -1, :]))
        generated_ids.append(next_token_id)

    return generated_ids