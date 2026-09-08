import numpy as np
import pandas as pd

def predict_new(model, districts):
    df = pd.DataFrame(districts)
    df = add_ratio_features(df)
    predictions = model.predict(df)
    return [float(np.round(pred)) for pred in predictions]