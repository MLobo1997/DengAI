from sklearn.base import BaseEstimator, TransformerMixin

class NoiseRemover(BaseEstimator, TransformerMixin):
    def __init__(self, noisy_weeks=53, copy=True):
        self.noisy_weeks = noisy_weeks
        self.train_set = True
        self.copy = copy

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.train_set:
            if self.copy:
                X = X.copy()
            X = X[X['weekofyear'] != self.noisy_weeks]
            self.train_set = False

        return X


