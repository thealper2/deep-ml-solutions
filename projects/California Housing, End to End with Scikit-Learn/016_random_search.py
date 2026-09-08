from sklearn.model_selection import RandomizedSearchCV

def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    param_distribs = {
        "columntransformer__geo__n_clusters": list(range(3, 11)),
        "randomforestregressor__max_features": list(range(2, 9)),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search