from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def stratified_regression_split(X, y, every_n, train_size, mean=False):
    # Create the bins. Given that y is not uniformly distributed at all (as you can see in the analysis notebook histogram), we created bins every 100 values.

    if mean:
        y2 = np.mean(y, axis=1)
    else:
        y2 = y.copy()

    y_sorted = np.sort(y2)
    bins = []
    for idx, val in enumerate(y_sorted):
        if (idx % every_n) == 0:
            bins.append(val)

    # Save your Y values in a new ndarray,
    # broken down by the bins created above.

    y_binned = np.digitize(y2, bins)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    for train_index, test_index in sss.split(X, y_binned):
        X_train_strat, X_test_strat = X[train_index], X[test_index]
        y_train_strat, y_test_strat = y[train_index], y[test_index]
        
    return X_train_strat, y_train_strat, X_test_strat, y_test_strat