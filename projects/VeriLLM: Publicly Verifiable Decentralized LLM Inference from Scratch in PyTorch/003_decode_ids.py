def decode_ids(ids, vocab):
    return ''.join(vocab['itos'][i] for i in ids)
