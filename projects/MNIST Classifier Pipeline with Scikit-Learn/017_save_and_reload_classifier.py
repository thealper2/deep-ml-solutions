import joblib

def save_and_reload_classifier(model, path):
    joblib.dump(model, path)
    return joblib.load(path)