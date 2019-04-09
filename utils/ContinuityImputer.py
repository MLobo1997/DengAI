from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ContinuityImputer(BaseEstimator, TransformerMixin):
    def __init__(self, attributes, copy=True):
        self.attributes = attributes
        self.copy = copy
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.copy:
            X = X.copy()

        for attr in self.attributes:
            last_values = {'sj': 0, 'iq': 0}
            r = []
            for _, curr in X.iterrows():
                city = curr['city']
                val = curr[attr]
                if val is not None and not np.isnan(val):
                    last_values[city] = val
                r.append(last_values[city])
            X[attr] = r

        return X