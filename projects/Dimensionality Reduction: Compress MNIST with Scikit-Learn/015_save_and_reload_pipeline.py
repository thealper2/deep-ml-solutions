import joblib

def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)