def split_score(parent_labels, left_labels, right_labels):
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    
    parent_impurity = impurity(parent_labels)
    weighted_child_impurity = (n_left / n) * impurity(left_labels) + (n_right / n) * impurity(right_labels)
    
    return parent_impurity - weighted_child_impurity
