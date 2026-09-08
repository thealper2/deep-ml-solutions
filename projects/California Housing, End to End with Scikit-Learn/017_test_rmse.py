def test_rmse(model, test_set):
    X_test, y_test = split_features_labels(add_ratio_features(test_set))
    y_pred = model.predict(X_test)
    return rmse(y_test, y_pred)