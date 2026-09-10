from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier_on_compressed(X, y, n_components, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    raw_clf = LogisticRegression(max_iter=2000)
    raw_clf.fit(X_train, y_train)
    raw_accuracy = float(raw_clf.score(X_test, y_test))

    pca = fit_pca(X_train, n_components)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_clf = LogisticRegression(max_iter=2000)
    pca_clf.fit(X_train_pca, y_train)
    pca_accuracy = float(pca_clf.score(X_test_pca, y_test))

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (X_train.shape[1], X_train_pca.shape[1]),
    }