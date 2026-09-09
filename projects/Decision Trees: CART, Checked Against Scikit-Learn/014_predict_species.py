import numpy as np

def predict_species(clf, measurements, target_names):
    X = np.array(measurements)
    preds = clf.predict(X)
    return [str(target_names[int(p)]) for p in preds]