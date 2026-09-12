from sklearn.linear_model import SGDClassifier

def train_sgd(X, y, random_state=42):
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf