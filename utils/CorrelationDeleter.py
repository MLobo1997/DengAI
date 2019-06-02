from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CorrelationDeleter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_corr=0.9):
        self.threshold_corr=threshold_corr
    
    def fit(self, X, y=None):
        corr_matrix = np.corrcoef(X, rowvar=False)
        self.drop_cols = []
        n_cols = corr_matrix.shape[1]
        for i in range(n_cols - 1):
            for j in range(i+1,n_cols):
                if j not in self.drop_cols:
                    val = corr_matrix[i, j]
                    if abs(val) >= self.threshold_corr:
                        self.drop_cols.append(j) 
        return self
    
    def transform(self, X):
        return np.delete(X, self.drop_cols, axis=1)
