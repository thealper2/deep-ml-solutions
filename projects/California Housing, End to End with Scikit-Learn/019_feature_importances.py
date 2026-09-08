def feature_importances(search, k=5):
    best_estimator = search.best_estimator_
    preprocessor = best_estimator.steps[0][1]
    feature_names = preprocessor.get_feature_names_out()
    importances = best_estimator.steps[-1][1].feature_importances_
    pairs = sorted(zip(importances, feature_names), reverse=True)[:k]
    return [(float(round(imp, 3)), name) for imp, name in pairs]