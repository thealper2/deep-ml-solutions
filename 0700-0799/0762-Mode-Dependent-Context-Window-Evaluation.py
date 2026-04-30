def prepare_eval_input(tokens: list, mode: str, reserved_output: int) -> list:
    """
    Truncate a token list to fit the context window of the given reasoning mode.
    """
    context_windows = {
        'non-think': 8192,
        'high': 131072,
        'max': 393216,
    }

    if mode not in context_windows:
        raise ValueError('Invalid mode.')

    C = context_windows[mode]
    L = C - reserved_output

    if L <= 0:
        return []

    if len(tokens) > L:
        return tokens[-L:]
    else:
        return tokens[:]