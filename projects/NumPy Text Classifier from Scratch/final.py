"""
NumPy Text Classifier from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  clean_text ──
def clean_text(text: str) -> str:
    text = text.lower()
    result = ''.join(c if c.isalpha() else ' ' for c in text)
    return result.rstrip()

# ── Step 002  tokenize ──
def tokenize(text: str) -> list:
    return [word.strip() for word in text.split()]

# ── Step 003  tokenize_corpus ──
def tokenize_corpus(texts: list) -> list:
    return [tokenize(clean_text(text)) for text in texts]

# ── Step 004  split_train_val_test_indices ──
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:n_train+n_val+n_test]
    return train_idx, val_idx, test_idx

# ── Step 005  count_word_frequencies ──
def count_word_frequencies(tokenized_docs: list) -> dict:
    freq = {}
    for doc in tokenized_docs:
        for word in doc:
            freq[word] = freq.get(word, 0) + 1

    return freq

# ── Step 006  build_vocabulary ──
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    top_words = sorted_words[:max_size]
    vocab = {word: idx for idx, (word, _) in enumerate(top_words)}
    return vocab

# ── Step 007  tokens_to_bow ──
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    V = len(vocab)
    bow = np.zeros(V, dtype=float)
    for token in tokens:
        if token in vocab:
            bow[vocab[token]] += 1.0
            
    return bow

# ── Step 008  corpus_to_bow_matrix ──
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    N = len(tokenized_docs)
    V = len(vocab)
    bow = np.zeros((N, V), dtype=float)

    for i, doc in enumerate(tokenized_docs):
        for token in doc:
            if token in vocab:
                bow[i, vocab[token]] += 1.0

    return bow

# ── Step 009  compute_document_frequencies ──
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    return np.sum(bow_matrix > 0, axis=0)

# ── Step 010  compute_idf ──
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    return np.log((n_docs + 1) / (df + 1)) + 1

# ── Step 011  transform_tfidf ──
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    return bow_matrix * idf

# ── Step 012  fit_tfidf ──
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]
    idf = compute_idf(df, n_docs)
    return idf

# ── Step 013  sigmoid ──
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    out[~positive] = np.exp(z[~positive]) / (1.0 + np.exp(z[~positive]))
    return out

# ── Step 014  logistic_predict_proba ──
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    out = sigmoid(z)
    return out

# ── Step 015  binary_cross_entropy ──
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    bce = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    loss = bce + l2_lambda * np.sum(w ** 2) / 2
    return float(loss)

# ── Step 016  logistic_gradients ──
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    r = y_prob - y_true
    term = (X.T @ r) / len(r)
    dw = term +  l2_lambda * w
    db = np.mean(r)
    return dw, db

# ── Step 017  initialize_logistic_params ──
def initialize_logistic_params(n_features: int) -> tuple:
    w = np.zeros(n_features)
    b = 0.0
    return w, b

# ── Step 018  gradient_descent_step ──
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    m = X.shape[0]
    probs = logistic_predict_proba(X, w, b)
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1 - eps)
    loss = binary_cross_entropy(y, probs_clipped, w, l2_lambda)
    dw = (1 / m) * X.T @ (probs - y) + l2_lambda * w
    db = (1 / m) * np.sum(probs - y)
    w_new = w - lr * dw
    b_new = b - lr * db
    
    return w_new, b_new, loss

# ── Step 019  train_logistic_regression ──
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    N, D = X.shape
    w, b = initialize_logistic_params(D)
    losses = []
    
    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)
    
    return w, b, losses

# ── Step 020  predict_labels ──
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    return np.where(proba >= threshold, 1, 0)

# ── Step 021  confusion_counts ──
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn

# ── Step 022  metrics_from_counts ──
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }

# ── Step 023  tune_decision_threshold ──
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = 0.5
    best_f1 = -1.0
    
    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# ── Step 024  evaluate_predictions ──
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['recall'],
        'accuracy': metrics['accuracy'],
    }

# ── Step 025  vectorize_texts ──
def vectorize_texts(texts: list, vocab: dict, idf: np.ndarray) -> np.ndarray:
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)
    tfidf_matrix = transform_tfidf(bow_matrix, idf)
    return tfidf_matrix

# ── Step 026  predict_text ──
def predict_text(text: str, vocab: dict, idf: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> int:
    """Label a single raw message with the fitted classifier.

    Args:
        text: Raw input string.
        vocab: Fitted word -> column index map.
        idf: Fitted IDF vector, shape (V,).
        w: Logistic weight vector, shape (V,).
        b: Logistic bias scalar.
        threshold: Decision threshold for the positive class.

    Returns:
        Predicted label as int 0 or 1.
    """
    x = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(x, w, b)
    preds = predict_labels(proba, threshold)
    return int(preds[0])

# ── Step 027  collect_prediction_errors ──
def collect_prediction_errors(texts: list, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    false_positives = []
    false_negatives = []
    for text, y_t, y_p in zip(texts, y_true, y_pred):
        if y_t == 1 and y_p == 0:
            false_negatives.append(text)
        elif y_t == 0 and y_p == 1:
            false_positives.append(text)

    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }

# ── Scaffold (runner) ──
"""End-to-end demo: NumPy TF-IDF + L2 logistic spam/sentiment classifier."""
import numpy as np


def main() -> None:
    np.random.seed(0)

    texts = [
        "Win a free prize now click here",
        "Meeting scheduled for Monday morning",
        "Congratulations you won the lottery jackpot",
        "Please review the attached project report",
        "Cheap meds online limited offer today",
        "Lunch with the team at noon tomorrow",
        "Urgent claim your reward immediately",
        "Can we reschedule the client call",
        "Free money guaranteed no risk act now",
        "Notes from yesterday standup are ready",
        "Exclusive deal buy one get one free",
        "The quarterly budget looks healthy",
        "You have been selected for a gift card",
        "Draft agenda for the all hands meeting",
        "Click to unsubscribe and win cash",
        "Ship the release candidate this Friday",
    ]
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

    tokenized = tokenize_corpus(texts)
    n = len(texts)
    train_idx, val_idx, test_idx = split_train_val_test_indices(
        n, val_fraction=0.25, test_fraction=0.25, seed=0
    )

    train_tok = [tokenized[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    word_counts = count_word_frequencies(train_tok)
    vocab = build_vocabulary(word_counts, max_size=40)
    bow_train = corpus_to_bow_matrix(train_tok, vocab)
    idf = fit_tfidf(bow_train)
    X_train = transform_tfidf(bow_train, idf)

    w, b, losses = train_logistic_regression(
        X_train, y_train, lr=0.5, l2_lambda=0.01, n_epochs=80
    )
    print("vocab_size", len(vocab))
    print("final_train_loss", round(float(losses[-1]), 4))

    X_val = vectorize_texts(val_texts, vocab, idf)
    val_proba = logistic_predict_proba(X_val, w, b)
    best_t, best_f1 = tune_decision_threshold(y_val, val_proba)
    print("best_threshold", round(float(best_t), 3), "val_f1", round(float(best_f1), 3))

    X_test = vectorize_texts(test_texts, vocab, idf)
    test_proba = logistic_predict_proba(X_test, w, b)
    y_pred = predict_labels(test_proba, threshold=best_t)
    report = evaluate_predictions(y_test, y_pred)
    print("test_metrics", {k: round(v, 3) if isinstance(v, float) else v for k, v in report.items()})

    sample = "Free prize click here to claim now"
    pred = predict_text(sample, vocab, idf, w, b, threshold=best_t)
    print("sample_text", sample)
    print("sample_pred", int(pred))

    errors = collect_prediction_errors(test_texts, y_test, y_pred)
    print("n_false_positives", len(errors["false_positives"]))
    print("n_false_negatives", len(errors["false_negatives"]))


if __name__ == "__main__":
    main()
