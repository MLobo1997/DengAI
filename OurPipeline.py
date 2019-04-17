from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.ContinuityImputer import ContinuityImputer
from utils.DataFrameDropper import DataFrameDropper
from utils.LastWeeks import LastWeeks
from utils.LastInfected import LastInfected

def create_pipeline(attr, n_weeks, pca_n_components=None,  n_non_train=4):
    pipelist = [
        ('imputer', ContinuityImputer(attributes=attr[n_non_train:])),
        ('lw', LastWeeks(attributes=attr[n_non_train:], weeks=n_weeks)),
        ('lf', LastInfected(weeks=n_weeks)),
        ('dataframe_dropper', DataFrameDropper(attribute_names=attr[:n_non_train])),
        ('scaler', StandardScaler()),
    ]

    if pca_n_components is not None:
        pipelist.append(('pca', PCA(n_components=pca_n_components)))

    return Pipeline(pipelist)