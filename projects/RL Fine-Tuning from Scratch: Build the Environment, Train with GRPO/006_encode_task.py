VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    ids = [BOS]
    for n in nums:
        ids.append(TOK[str(n)])

    ids.append(SEP)
    for ch in f"{target:03d}":
        ids.append(TOK[ch])

    return ids

def decode_completion(ids):
    result = []
    for idx in ids:
        i = int(idx)
        if i == EOS:
            break

        result.append(VOCAB[i])
    
    return result