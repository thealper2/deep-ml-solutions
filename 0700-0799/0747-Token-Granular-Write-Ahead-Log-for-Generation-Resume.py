def resume_generation(prompt, wal, max_new_tokens, vocab_size, eos):
    """
    Resume deterministic token generation using a Write-Ahead Log.

    Args:
        prompt: list of ints, initial prompt tokens.
        wal: list of ints, tokens already generated before interruption.
        max_new_tokens: int, max total number of generated tokens (not counting prompt).
        vocab_size: int, modulus for the next-token rule.
        eos: int, end-of-sequence token id.

    Returns:
        list of ints: prompt followed by the full generated sequence.
    """
    if eos in wal:
        eos_idx = wal.index(eos)
        return prompt + wal[:eos_idx + 1]

    context = prompt + wal.copy()
    generated = wal.copy()
    tokens_generated = len(generated)

    while tokens_generated < max_new_tokens:
        if len(context) >= 2:
            next_token = (context[-1] + context[-2]) % vocab_size
        elif len(context) == 1:
            next_token = context[-1] % vocab_size
        else:
            next_token = 0

        context.append(next_token)
        generated.append(next_token)
        tokens_generated += 1

        if next_token == eos:
            break

    return prompt + generated
