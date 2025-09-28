from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=10, random_state=42):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_ = None
        self.maps_ = {}  # per-col mapping

    def get_feature_names_out(self, input_features=None):
    # TargetEncoder menghasilkan satu kolom per input feature,
    # jadi kita cukup mengembalikan nama kolom aslinya.
        return np.asarray(self.cols, dtype=object)

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y).astype(float)
        self.global_mean_ = y.mean()

        # full-data mapping (untuk transform pada test/infer)
        for c in self.cols:
            stats = y.groupby(X[c]).agg(['mean','count'])
            enc = (stats['count']*stats['mean'] + self.smoothing*self.global_mean_) / (stats['count']+self.smoothing)
            self.maps_[c] = enc.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].map(self.maps_.get(c, {})).fillna(self.global_mean_)
        return X
