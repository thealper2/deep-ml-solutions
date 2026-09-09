import joblib

def save_and_reload_tree(clf, path):
    joblib.dump(clf, path)
    return joblib.load(path)