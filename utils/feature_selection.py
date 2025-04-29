import numpy as np
import pandas as pd


def Pearson_correlation_coefficient(X, y):
    X = np.array(X, dtype=np.float64)
    y = np.array([y], dtype=np.float64).T
    data = pd.DataFrame(np.concatenate((X, y), axis=1))
    corr = data.corr(method='pearson').abs()
    feature_scores = corr.iloc[-1, 0:-1]

    num_features = corr.shape[0] - 1
    non_redundant_feats = np.full(num_features, 1, dtype=bool)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if corr.iloc[i, j] >= 0.90:
                if corr.iloc[i, num_features] < corr.iloc[j, num_features]:
                    non_redundant_feats[i] = 0
                else:
                    non_redundant_feats[j] = 0

    scores = np.multiply(feature_scores, non_redundant_feats)
    return scores
