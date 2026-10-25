"""The public tabular benchmarks of Section 8: loading and encoding.

Each table is represented by up to two encodings:

  E1  the numeric columns;
  E2  the numeric columns and the categorical columns as integer codes.

Rows with a missing value are dropped. Tables are downloaded from OpenML
(covertype through scikit-learn) on first use and cached by scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# cell tag -> (loader name, encoding)
CELLS = (
    ("adult:E1", "adult", "E1"),
    ("adult:E2", "adult", "E2"),
    ("covertype:E1", "covertype", "E1"),
    ("census-income:E1", "census-income", "E1"),
    ("bank-marketing:E1", "bank-marketing", "E1"),
    ("bank-marketing:E2", "bank-marketing", "E2"),
    ("higgs:E1", "higgs", "E1"),
    ("MagicTelescope:E1", "MagicTelescope", "E1"),
    ("pendigits:E1", "pendigits", "E1"),
    ("letter:E1", "letter", "E1"),
    ("shuttle:E1", "shuttle", "E1"),
)

# candidates excluded by the screening that precedes the analysis (Section 8.1)
EXCLUDED = (
    ("electricity:E1", "electricity", "E1"),
    ("eeg-eye-state:E1", "eeg-eye-state", "E1"),
    ("MiniBooNE:E1", "MiniBooNE", "E1"),
)


# OpenML identification of every source table except covertype
OPENML = {
    "adult": {"name": "adult", "version": 2},
    "census-income": {"name": "census-income", "version": 1},
    "bank-marketing": {"name": "bank-marketing", "version": 1},
    "higgs": {"data_id": 23512},
    "MagicTelescope": {"name": "MagicTelescope", "version": 1},
    "pendigits": {"data_id": 32},
    "letter": {"name": "letter", "version": 1},
    "shuttle": {"name": "shuttle", "version": 1},
    "electricity": {"data_id": 151},
    "eeg-eye-state": {"name": "eeg-eye-state", "version": 1},
    "MiniBooNE": {"data_id": 41150},
}


def _openml(source: str) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    return fetch_openml(**OPENML[source], as_frame=True, parser="auto").data


def _covertype() -> pd.DataFrame:
    from sklearn.datasets import fetch_covtype
    return pd.DataFrame(np.asarray(fetch_covtype(download_if_missing=True).data))


LOADERS = {source: (lambda s=source: _openml(s)) for source in OPENML}
LOADERS["covertype"] = _covertype


def numeric_mask(df: pd.DataFrame) -> pd.Series:
    out = []
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
            out.append(False)
        else:
            out.append(pd.api.types.is_numeric_dtype(s))
    return pd.Series(out, index=df.columns)


def encode_e1(df: pd.DataFrame) -> np.ndarray:
    """Numeric columns, complete rows."""
    num = df.loc[:, numeric_mask(df)]
    X = num.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return X[np.isfinite(X).all(axis=1)]


def encode_e2(df: pd.DataFrame) -> np.ndarray:
    """Numeric columns and categorical columns as integer codes, complete rows."""
    mask, cols = numeric_mask(df), []
    for c in df.columns:
        s = df[c]
        if mask[c]:
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
        else:
            codes, _ = pd.factorize(s, sort=True)
            codes = codes.astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
    X = np.column_stack(cols)
    return X[np.isfinite(X).all(axis=1)]


ENCODERS = {"E1": encode_e1, "E2": encode_e2}


class Tables:
    """Loads each source table once and returns encoded cells."""

    def __init__(self):
        self._cache = {}

    def frame(self, source: str) -> pd.DataFrame:
        if source not in self._cache:
            D = LOADERS[source]()
            self._cache[source] = (D.copy() if isinstance(D, pd.DataFrame)
                                   else pd.DataFrame(np.asarray(D)))
        return self._cache[source]

    def encoded(self, source: str, encoding: str) -> np.ndarray:
        return ENCODERS[encoding](self.frame(source))


def subsample(X: np.ndarray, max_rows: int, seed: int = 0) -> np.ndarray:
    """At most `max_rows` rows drawn without replacement, in original order."""
    if len(X) <= max_rows:
        return X
    g = np.random.default_rng(seed)
    return X[np.sort(g.choice(len(X), max_rows, replace=False))]


def law_from_record(rec: dict, n_rep: int, seed: int = 0):
    """The FloorLaw of a stored benchmark profile.

    The fit and the curvature test are rerun by
    certgen.design.floor_law_from_profile with the seed of
    benchmark_floor_laws.py, so the law reproduces the stored one.
    """
    from certgen.design import floor_law_from_profile
    se = rec.get("floor_se_log")
    return floor_law_from_profile(
        rec["n_grid"], rec["floors"], rec["floor_stds"],
        se_log=None if se is None else np.asarray(se, float),
        n_rep=n_rep, metric="w2", seed=seed)
