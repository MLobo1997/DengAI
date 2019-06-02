from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.ContinuityImputer import ContinuityImputer
from utils.DataFrameDropper import DataFrameDropper
from utils.LastWeeks import LastWeeks
from utils.LastInfected import LastInfected
from utils.CorrelationDeleter import CorrelationDeleter
from sklearn.feature_selection import SelectKBest, f_regression

def create_pipeline(attr, n_weeks=None, n_weeks_infected=None, estimator_optimizer=None, pca=None, add_noise=False, noise_mean=None, noise_std=None, n_non_train=4, k_best=None, threshold_corr=None):

    l_infected = None
    l_weeks = None
    sel_k_best = None
    corr_del = None
    if n_weeks_infected is not None and n_weeks_infected > 0:
        l_infected = LastInfected(weeks=n_weeks_infected, add_noise=add_noise, noise_mean=noise_mean, noise_std=noise_std)
    if n_weeks is not None and n_weeks > 0:
        l_weeks = LastWeeks(attributes=attr[n_non_train:], weeks=n_weeks)
    if k_best is not None:
        sel_k_best = SelectKBest(f_regression, k_best)
    if threshold_corr is not None:
        corr_del = CorrelationDeleter(threshold_corr=threshold_corr)

    return Pipeline([
        ('imputer', ContinuityImputer(attributes=attr[n_non_train:])),
        ('l_weeks', l_weeks),
        ('l_infected', l_infected),
        ('dataframe_dropper', DataFrameDropper(attribute_names=attr[:n_non_train])),
        ('scaler', StandardScaler()),
        ('pca', pca),
        ('sel_k_best', sel_k_best),
        ('corr_del', corr_del),
        ('est_opt', estimator_optimizer),
    ]
)