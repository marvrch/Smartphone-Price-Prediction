# encoders.py
from __future__ import annotations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    K-Fold Target Encoding dengan smoothing.
    - fit_transform(X, y): OOF encoding untuk TRAIN (anti-leakage)
    - fit(X, y): simpan mapping FULL-TRAIN untuk inference
    - transform(X): encode pakai mapping; kategori baru -> global mean
    """

    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        # Penting: JANGAN ubah/meng-cast parameter di sini (agar bisa di-clone)
        self.cols = cols                 # bisa list/tuple berisi nama atau posisi
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

        # atribut yang diisi saat fit
        self.global_mean_ = None
        self.maps_ = None                # {col_name: {category: encoded}}
        self._resolved_cols_ = None      # nama kolom aktual yang dipakai internal

    def get_feature_names_out(self, input_features=None):
        # output TE: satu kolom per input col
        # gunakan self.cols agar nama konsisten dengan yang diharapkan user
        return np.asarray(list(self.cols), dtype=object)

    # ---------- utilities ----------
    def _resolve_cols(self, X: pd.DataFrame):
        # Samakan 'self.cols' (nama/posisi) dengan kolom aktual di X
        xcols = list(X.columns)
        cols_seq = list(self.cols)

        # case: posisi integer
        if all(isinstance(c, (int, np.integer)) for c in cols_seq):
            idx = [int(c) for c in cols_seq]
            if max(idx) >= len(xcols):
                raise IndexError(f"cols index out of range: max {max(idx)} >= n_cols {len(xcols)}")
            resolved = [xcols[i] for i in idx]
            self._resolved_cols_ = resolved
            return resolved

        # case: nama string dan tersedia di X
        if set(cols_seq).issubset(set(xcols)):
            self._resolved_cols_ = cols_seq
            return cols_seq

        # fallback: jika jumlah sama, rename sementara
        if len(cols_seq) == len(xcols):
            self._resolved_cols_ = cols_seq
            return cols_seq

        raise KeyError(
            f"Cannot resolve cols. Incoming columns={xcols[:5]}...(n={len(xcols)}) "
            f"but encoder expects {cols_seq}"
        )

    def _to_frame(self, X):
        # Pastikan kita selalu bekerja dengan DataFrame + nama kolom yang sesuai
        if isinstance(X, pd.DataFrame):
            resolved = self._resolve_cols(X)
            df = X.copy()
            if list(df.columns) != resolved and len(df.columns) == len(resolved):
                df.columns = resolved
            return df.loc[:, resolved]

        # ndarray / Series -> bungkus jadi DataFrame
        df = pd.DataFrame(X)
        cols_seq = list(self.cols)
        if df.shape[1] != len(cols_seq):
            raise ValueError(f"Array has {df.shape[1]} cols but encoder expects {len(cols_seq)}")
        df.columns = cols_seq
        self._resolved_cols_ = cols_seq
        return df

    def _smoothed(self, stats: pd.DataFrame, mu: float) -> pd.Series:
        # stats: index=kategori, columns=['mean','count']
        s = float(self.smoothing)
        return (stats['count'] * stats['mean'] + s * mu) / (stats['count'] + s)

    # ---------- OOF TRAIN ----------
    def fit_transform(self, X, y):
        X = self._to_frame(X)
        y = pd.Series(y, index=X.index).astype(float)

        self.global_mean_ = float(y.mean())
        X_enc = pd.DataFrame(index=X.index, columns=self._resolved_cols_, dtype=float)

        kf = KFold(n_splits=int(self.n_splits), shuffle=True, random_state=self.random_state)
        for c in self._resolved_cols_:
            for tr_idx, val_idx in kf.split(X):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                stats = y_tr.groupby(X_tr[c]).agg(['mean', 'count'])
                enc_map = self._smoothed(stats, self.global_mean_)
                mapped = X.iloc[val_idx][c].map(enc_map).fillna(self.global_mean_)
                # assign pakai LABEL index (bukan posisi) → aman untuk index acak
                X_enc.loc[mapped.index, c] = mapped.to_numpy()

        # mapping FULL TRAIN utk inference
        self.maps_ = {}
        for c in self._resolved_cols_:
            stats_full = y.groupby(X[c]).agg(['mean', 'count'])
            self.maps_[c] = self._smoothed(stats_full, self.global_mean_).to_dict()

        # nama output konsisten dgn parameter cols
        X_enc.columns = list(self.cols)
        return X_enc

    # ---------- FIT (untuk inference) ----------
    def fit(self, X, y):
        X = self._to_frame(X)
        y = pd.Series(y, index=X.index).astype(float)

        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        for c in self._resolved_cols_:
            stats_full = y.groupby(X[c]).agg(['mean', 'count'])
            self.maps_[c] = self._smoothed(stats_full, self.global_mean_).to_dict()
        return self

    # ---------- TRANSFORM (test/Streamlit) ----------
    def transform(self, X):
        X = self._to_frame(X)
        out = pd.DataFrame(index=X.index, columns=self._resolved_cols_, dtype=float)
        for c in self._resolved_cols_:
            out[c] = X[c].map(self.maps_.get(c, {})).fillna(self.global_mean_)
        out.columns = list(self.cols)
        return out
