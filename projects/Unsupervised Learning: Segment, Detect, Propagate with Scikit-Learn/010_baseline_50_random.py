from sklearn.linear_model import LogisticRegression

def baseline_50_random(X_train, y_train, X_test, y_test, n_labeled=50, random_state=42):
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train[:n_labeled], y_train[:n_labeled])
    return float(lr.score(X_test, y_test))
