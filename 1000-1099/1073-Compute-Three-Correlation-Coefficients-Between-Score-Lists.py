import numpy as np
from scipy import stats

def correlations(x, y):
    pearson, _ = stats.pearsonr(x, y)
    spearman, _ = stats.spearmanr(x, y)
    kendall, _ = stats.kendalltau(x, y)

    return [pearson, spearman, kendall]
    
