from sklearn.tree import export_text

def tree_rules(clf, feature_names):
    return export_text(clf, feature_names=list(feature_names))