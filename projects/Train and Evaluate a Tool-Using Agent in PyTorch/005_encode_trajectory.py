def encode_trajectory(text, block_size=40):
    """Token ids and a 0/1 loss mask (1 = written by the model), both padded to block_size."""
    tokens = text.split()
    if len(tokens) > block_size:
        raise ValueError("trajectory too long")

    mask = []
    spans = [("<call>", "</call>"), ("<answer>", "</answer>")]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        opened = False
        for start, end in spans:
            if tok == start:
                mask.append(1)
                j = i + 1
                while j < len(tokens) and tokens[j] != end:
                    mask.append(1)
                    j += 1
                if j < len(tokens):
                    mask.append(1)
                    i = j + 1
                else:
                    i = j
                opened = True
                break
        if opened:
            continue
        mask.append(0)
        i += 1

    ids = [STOI.get(t, STOI["<unk>"]) for t in tokens]

    pad_id = STOI["<pad>"]
    while len(ids) < block_size:
        ids.append(pad_id)
        mask.append(0)

    return ids, mask