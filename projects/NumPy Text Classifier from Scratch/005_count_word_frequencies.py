def count_word_frequencies(tokenized_docs: list) -> dict:
    freq = {}
    for doc in tokenized_docs:
        for word in doc:
            freq[word] = freq.get(word, 0) + 1

    return freq
