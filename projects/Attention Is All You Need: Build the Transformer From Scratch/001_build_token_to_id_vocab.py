def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    i = 0
    stoi = {}
    for special in specials:
        stoi[special] = i
        i += 1

    for sentence in sentences:
        for word in sentence.split():
            if word not in stoi.keys():
                stoi[word] = i
                i += 1

    return stoi
