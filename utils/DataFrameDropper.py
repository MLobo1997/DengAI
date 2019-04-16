from sklearn.base import BaseEstimator, TransformerMixin
import pandas

class DataFrameDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names, copy = True):
        self.attribute_names = attribute_names
        self.copy = copy
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()
        if isinstance(X, pandas.core.frame.DataFrame):
            return X.drop(self.attribute_names, axis = 1)

        raise ValueError('You try to drop some columns from something which is not a DataFrame')