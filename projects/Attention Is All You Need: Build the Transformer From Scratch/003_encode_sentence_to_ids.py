def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    result = []
    for word in sentence.split():
        idx = token_to_id.get(word, token_to_id[unk_token])
        result.append(idx)

    return result
