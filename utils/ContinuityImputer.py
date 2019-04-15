from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ContinuityImputer(BaseEstimator, TransformerMixin):
    def __init__(self, attributes, copy=True):
        self.attributes = attributes
        self.copy = copy
    
    def fit(self, X, y=None):
        X_iq = X[X['city'] == 'iq']
        X_sj = X[X['city'] == 'sj']

        medians_iq = {attr: np.nanmedian(X_iq[attr]) for attr in self.attributes}
        medians_sj = {attr: np.nanmedian(X_sj[attr]) for attr in self.attributes}
        self.last_values = {'sj': medians_sj, 'iq': medians_iq}

        return self
    
    def transform(self, X):
        if self.copy:
            X = X.copy()

        for attr in self.attributes:
            r = []
            for _, curr in X.iterrows():
                city = curr['city']
                val = curr[attr]
                if val is not None and not np.isnan(val):
                    self.last_values[city][attr] = val
                r.append(self.last_values[city][attr])
            X[attr] = r

        return X