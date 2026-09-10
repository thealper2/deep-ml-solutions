from sklearn.linear_model import LogisticRegression

def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])
    return float(model.score(X_test, y_test))