def custom_collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_batch = []
    targets_batch = []

    for item in batch:
        padded = item + [pad_token_id]

        while len(padded) < batch_max_length:
            padded.append(pad_token_id)

        inputs = padded[:-1]
        targets = padded[1:]

        found_first_pad = False
        for i in range(len(targets)):
            if targets[i] == pad_token_id:
                if not found_first_pad:
                    found_first_pad = True
                else:
                    targets[i] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_batch.append(inputs)
        targets_batch.append(targets)

    return {'inputs': inputs_batch, 'targets': targets_batch}
