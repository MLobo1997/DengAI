from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.ContinuityImputer import ContinuityImputer
from utils.DataFrameDropper import DataFrameDropper
from utils.LastWeeks import LastWeeks
from utils.LastInfected import LastInfected

def create_pipeline(attr, n_weeks, n_weeks_infected=None, estimator_optimizer=None, pca=None, add_noise=False, noise_mean=None, noise_std=None, n_non_train=4):

    l_infected = None
    if n_weeks_infected is not None and n_weeks_infected > 0:
        l_infected = LastInfected(weeks=n_weeks_infected, add_noise=add_noise, noise_mean=noise_mean, noise_std=noise_std)

    return Pipeline([
        ('imputer', ContinuityImputer(attributes=attr[n_non_train:])),
        ('l_weeks', LastWeeks(attributes=attr[n_non_train:], weeks=n_weeks)),
        ('l_infected', l_infected),
        ('dataframe_dropper', DataFrameDropper(attribute_names=attr[:n_non_train])),
        ('scaler', StandardScaler()),
        ('pca', pca),
        ('est_opt', estimator_optimizer),
    ]
)