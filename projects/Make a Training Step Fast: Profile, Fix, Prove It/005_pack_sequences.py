def pack_sequences(lengths, max_len):
    """First-fit decreasing packing into rows of max_len, with offsets and padding shares."""
    n = len(lengths)
    cut = [min(L, max_len) for L in lengths]

    order = sorted(range(n), key=lambda i: (-cut[i], i))

    rows = []
    row_used = []
    for i in order:
        placed = False
        for r in range(len(rows)):
            if row_used[r] + cut[i] <= max_len:
                rows[r].append(i)
                row_used[r] += cut[i]
                placed = True
                break
        if not placed:
            rows.append([i])
            row_used.append(cut[i])
        
    cu_seqlens = []
    for row in rows:
        cu = [0]
        for i in row:
            cu.append(cu[-1] + cut[i])
        cu_seqlens.append(cu)

    tokens_before = n * max_len
    tokens_after = len(rows) * max_len

    real_before = sum(cut)
    real_after = sum(cut)

    pad_share_before = (tokens_before - real_before) / tokens_before if tokens_before > 0 else 0.0
    pad_share_after = (tokens_after - real_after) / tokens_after if tokens_after > 0 else 0.0

    return {
        "rows": rows,
        "cu_seqlens": cu_seqlens,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "pad_share_before": pad_share_before,
        "pad_share_after": pad_share_after,
    }