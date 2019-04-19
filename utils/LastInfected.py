from sklearn.base import BaseEstimator, TransformerMixin
from collections import deque
import numpy as np
import pandas as pd
from random import gauss, choice

class LastInfected(BaseEstimator, TransformerMixin):
    def __init__(self, weeks=1, new_attributes_prefix='last_infected_', add_noise=False, noise_mean=None, noise_std=None, copy=True):
        self.weeks=weeks
        self.new_attributes_prefix = new_attributes_prefix
        self.copy=copy
        dq = deque([0 for _ in range(weeks)])
        self.last = {'sj': dq.copy(), 'iq': dq.copy()}
        self.add_noise = add_noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.first = True
    
    def fit(self, X, y):
        self.y = y.to_list()
        return self
    
    def transform(self, X, model=None):
        if self.copy:
            X = X.copy()
        
        X.reset_index(drop=True, inplace=True)

        r = np.ndarray(shape=[X.shape[0], self.weeks])

        for idx, x in X.iterrows():
            self.city = x['city']
            r[idx] = self.last[self.city]
            if self.first:
                self.append_y(self.y[idx])

        r = pd.DataFrame(r, columns=[self.new_attributes_prefix + str(week) for week in range(self.weeks)])
        
        X = pd.concat([X, r], axis=1)

        self.first=False

        return X

    def append_y(self, new_y):
        if self.add_noise:
            noise = int(np.round(choice([-1,1]) * gauss(mu=self.noise_mean, sigma=self.noise_std)))
            new_y += noise
            #if new_y < 0:
                #new_y = 0
        self.last[self.city].appendleft(new_y)
        self.last[self.city].pop()