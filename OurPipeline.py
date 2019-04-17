from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.ContinuityImputer import ContinuityImputer
from utils.DataFrameDropper import DataFrameDropper
from utils.LastWeeks import LastWeeks
from utils.LastInfected import LastInfected

def create_pipeline(attr, n_weeks, n_weeks_infected, estimator_optimizer=None, pca=None, n_non_train=4):

    return Pipeline([
        ('imputer', ContinuityImputer(attributes=attr[n_non_train:])),
        ('l_weeks', LastWeeks(attributes=attr[n_non_train:], weeks=n_weeks)),
        ('l_infected', LastInfected(weeks=n_weeks_infected)),
        ('dataframe_dropper', DataFrameDropper(attribute_names=attr[:n_non_train])),
        ('scaler', StandardScaler()),
        ('pca', pca),
        ('est_opt', estimator_optimizer),
    ]
)