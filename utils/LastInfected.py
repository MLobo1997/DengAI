from sklearn.base import BaseEstimator, TransformerMixin
from collections import deque
import numpy as np
import pandas as pd

class LastInfected(BaseEstimator, TransformerMixin):
    def __init__(self, weeks=1, new_attributes_prefix='last_infected_', copy=True):
        self.weeks=weeks
        self.new_attributes_prefix = new_attributes_prefix
        self.copy=copy
        dq = deque([0 for _ in range(weeks)])
        self.last = {'sj': dq.copy(), 'iq': dq.copy()}
    
    def fit(self, X, y):
        self.y = y
        return self
    
    def transform(self, X, model=None):
        if self.copy:
            X = X.copy()
        
        r = np.ndarray(shape=[X.shape[0], self.weeks])

        for idx, n_infected in enumerate(self.y):
            city = X.loc[idx, 'city']
            r[idx] = self.last[city]
            self.last[city].appendleft(n_infected)
            self.last[city].pop()

        r = pd.DataFrame(r, columns=[self.new_attributes_prefix + str(week) for week in range(self.weeks)])
        
        X = pd.concat([X, r], axis=1)

        return X