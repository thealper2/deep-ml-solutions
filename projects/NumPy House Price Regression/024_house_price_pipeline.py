def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)
    X_full = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels)
    splits = make_train_val_test(X_full, y, train_ratio, val_ratio, seed)
    std_splits, mean, std = standardize_and_add_bias(splits)
    theta = ols_fit(std_splits['X_train'], std_splits['y_train'])
    y_val_pred = ols_predict(std_splits['X_val'], theta)
    y_test_pred = ols_predict(std_splits['X_test'], theta)
    val_metrics = evaluate_predictions(std_splits['y_val'], y_val_pred)
    test_metrics = evaluate_predictions(std_splits['y_test'], y_test_pred)
    
    return {
        'theta': theta,
        'y_test': std_splits['y_test'],
        'y_test_pred': y_test_pred,
        'test_metrics': test_metrics,
        'val_metrics': val_metrics
    }
