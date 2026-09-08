import joblib

def save_and_reload(model, path):
    joblib.dump(model, path)
    return joblib.load(path)