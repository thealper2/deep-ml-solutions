import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(numerator, denominator)
    X_extended = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_encoded = one_hot_encode(cat_labels)
        X_extended = np.hstack([X_extended, cat_encoded])

    return X_extended
