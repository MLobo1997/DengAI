from sklearn.base import BaseEstimator, TransformerMixin
from collections import deque
import numpy as np
import pandas as pd


class LastWeeks(BaseEstimator, TransformerMixin):
    def __init__(self, attributes, weeks=2, new_attributes_prefix='last_weeks_', copy=True):
        self.attributes = attributes
        self.weeks = weeks
        self.new_attributes_prefix = new_attributes_prefix
        self.copy = copy

    def fit(self, X, y=None):
        attr_medians = [np.nanmedian(X[attr]) for attr in self.attributes]
        dq = deque([attr_medians for _ in range(self.weeks)])
        self.last = {'sj': dq, 'iq': dq}

        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()

        r = np.ndarray(shape=[X.shape[0], self.weeks, len(self.attributes)])

        for idx, (_, week) in enumerate(X.iterrows()):
            city = week['city']
            r[idx] = self.last[city]
            self.last[city].pop()
            self.last[city].appendleft(week[self.attributes])

        r = pd.DataFrame(r.reshape([X.shape[0], self.weeks * len(self.attributes)]),
                     columns=[self.new_attributes_prefix + str(week) + '_' + str(attr)
                              for week in range(self.weeks)
                              for attr in self.attributes
                              ])
        
        X = pd.concat([X, r], axis=1)

        return X
