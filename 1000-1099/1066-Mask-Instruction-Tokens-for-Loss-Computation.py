def mask_instruction_loss(batch, pad_token_id, ignore_index=-100):
    max_len = max(len(item['tokens']) for item in batch)
    inputs_batch = []
    targets_batch = []

    for item in batch:
        tokens = item['tokens']
        instruction_len = item['instruction_length']
        padded = tokens + [pad_token_id] * (max_len - len(tokens))
        inputs = padded[:-1]
        targets = padded[1:]
        for i in range(len(targets)):
            if i + 1 < instruction_len:
                targets[i] = ignore_index

            first_pad_idx = None
            for idx, val in enumerate(padded):
                if val == pad_token_id:
                    first_pad_idx = idx
                    break

            if first_pad_idx is not None and i + 1 > first_pad_idx:
                targets[i] = ignore_index

        inputs_batch.append(inputs)
        targets_batch.append(targets)

    return {'inputs': inputs_batch, 'targets': targets_batch}