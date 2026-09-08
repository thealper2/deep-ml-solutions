def worst_errors(model, df, k=5):
    X_test, y_test = split_features_labels(add_ratio_features(df))
    y_pred = model.predict(X_test)
    
    errors = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'abs_error': np.abs(y_test - y_pred)
    }, index=df.index)
    
    return errors.nlargest(k, 'abs_error')