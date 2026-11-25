SEQ_LENS = [4096, 16384, 65536, 1048576]


def _stage(tokens_seen, stage_boundaries):
    stage = 0
    for b in stage_boundaries:
        if tokens_seen >= b:
            stage += 1
    return stage

def attention_curriculum(tokens_seen, stage_boundaries, model="flash",
                         dense_tokens=1_000_000_000_000,
                         indexer_warmup_tokens=10_000_000_000):
    """
    Step-indexed attention curriculum of DeepSeek-V4 (report section 4.2.2).

    tokens_seen:            non-negative int, training tokens consumed so far
    stage_boundaries:       [b1, b2, b3] token counts at which the sequence length moves 4K->16K, 16K->64K, 64K->1M
    model:                  "flash" (top-k 512) or "pro" (top-k 1024)
    dense_tokens:           dense-attention budget (1T in the report)
    indexer_warmup_tokens:  length of the lightning-indexer warmup stage

    Returns {"stage", "seq_len", "attention", "indexer_warmup", "top_k"}.
    """
    stage = _stage(tokens_seen, stage_boundaries)
    seq_len = SEQ_LENS[stage]

    b2 = stage_boundaries[1]
    sparse_start = max(dense_tokens, b2)
    full_sparse = sparse_start + indexer_warmup_tokens

    if tokens_seen < sparse_start:
        attention = "dense"
        indexer_warmup = False
        top_k = None
    elif tokens_seen < full_sparse:
        attention = "dense"
        indexer_warmup = True
        top_k = None
    else:
        attention = "sparse"
        indexer_warmup = False
        top_k = 512 if model == "flash" else 1024

    return {
        "stage": stage,
        "seq_len": seq_len,
        "attention": attention,
        "indexer_warmup": indexer_warmup,
        "top_k": top_k,
    }


def curriculum_timeline(token_counts, stage_boundaries, model="flash",
                        dense_tokens=1_000_000_000_000,
                        indexer_warmup_tokens=10_000_000_000):
    """
    List of (seq_len, label) for every token count, label in {"dense", "indexer_warmup", "sparse"}.
    """
    out = []
    for tokens_seen in token_counts:
        r = attention_curriculum(tokens_seen, stage_boundaries, model,
                                 dense_tokens, indexer_warmup_tokens)
        if r["indexer_warmup"]:
            label = "indexer_warmup"
        elif r["attention"] == "sparse":
            label = "sparse"
        else:
            label = "dense"
        out.append((r["seq_len"], label))
    return out
