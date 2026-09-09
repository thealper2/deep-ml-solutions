import numpy as np

def flag_anomalies(gmm, X, contamination=0.04):
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)
    return densities < threshold