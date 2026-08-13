import numpy as np

class SentimentClassifier:
    def __init__(self, lr=0.5, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.vocab = {}
        self.idf = None
        self.w = None
        self.b = 0.0

    def _tokenize(self, text):
        return text.lower().split()

    def _build_vocab(self, texts):
        vocab = {}
        for text in texts:
            for token in self._tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)

        return vocab

    def _features(self, text, vocab, idf=None):
        tokens = self._tokenize(text)
        vec = np.zeros(len(vocab))
        token_counts = {}
        for token in tokens:
            if token in vocab:
                token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            idx = vocab[token]
            if idf is not None:
                vec[idx] = count * idf[idx]
            else:
                vec[idx] = count

        return vec

    def _compute_idf(self, texts, vocab):
        N = len(texts)
        doc_freq = np.zeros(len(vocab))
        for text in texts:
            tokens_seen = set(self._tokenize(text))
            for token in tokens_seen:
                if token in vocab:
                    doc_freq[vocab[token]] += 1

        idf = np.zeros(len(vocab))
        for i, df in enumerate(doc_freq):
            if df > 0:
                idf[i] = np.log(N / df) + 1

        return idf

    def fit(self, texts, labels):
        self.vocab = self._build_vocab(texts)
        vocab_size = len(self.vocab)

        self.idf = self._compute_idf(texts, self.vocab)

        X = np.array([self._features(text, self.vocab, self.idf) for text in texts])
        y = np.array(labels)

        self.w = np.zeros(vocab_size)
        self.b = 0.0

        n_samples = len(X)
        for epoch in range(self.epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for i in range(n_samples):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                z = np.dot(x_i, self.w) + self.b
                pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

                grad_w = x_i * (pred - y_i)
                grad_b = pred - y_i

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, text):
        if self.w is None:
            raise ValueError("Model not trained yet")
        
        x = self._features(text, self.vocab, self.idf)
        z = np.dot(x, self.w) + self.b
        prob = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return 1 if prob >= 0.5 else 0
