def swa_cache_strategies(prefix_len, n_win, p, num_layers, entry_bytes):
    """
    Bytes written, bytes read on a prefix hit and tokens recomputed for the three on-disk
    SWA KV cache strategies of report section 3.5.2 (full, periodic, zero).
    per_token = num_layers * entry_bytes.
    Returns {"full": {...}, "periodic": {...}, "zero": {...}} with int values under the keys
    "bytes_written", "bytes_read", "tokens_recomputed".
    """
    per_token = num_layers * entry_bytes

    full = {
        "bytes_written": prefix_len * per_token,
        "bytes_read": min(n_win, prefix_len) * per_token,
        "tokens_recomputed": 0,
    }

    k = prefix_len // p
    if k == 0:
        periodic = {
            "bytes_written": 0,
            "bytes_read": 0,
            "tokens_recomputed": min(prefix_len, n_win * num_layers),
        }
    else:
        written = sum(min(n_win, j * p) for j in range(1, k + 1)) * per_token
        c = k * p
        read = min(n_win, c) * per_token
        recomputed = prefix_len - c
        periodic = {
            "bytes_written": written,
            "bytes_read": read,
            "tokens_recomputed": recomputed,
        }

    zero = {
        "bytes_written": 0,
        "bytes_read": 0,
        "tokens_recomputed": min(prefix_len, n_win * num_layers),
    }

    return {"full": full, "periodic": periodic, "zero": zero}