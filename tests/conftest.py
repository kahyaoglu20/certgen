"""Shared fixtures of the test suite."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("", "tools", "experiments", "tests"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import certgen as cg  # noqa: E402


def artifact(name: str):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def copula6():
    """Standardised six-dimensional copula, 4000 rows."""
    return cg.Standardizer().fit_transform(cg.data.gaussian_copula(4000, 6, seed=7))


@pytest.fixture(scope="session")
def law6(copula6):
    """Floor law of `copula6` (admissible)."""
    return cg.fit_floor_law(copula6, metric="w2", n_rep=10)


@pytest.fixture(scope="session")
def gaussian_law():
    """Floor law of a five-dimensional standard normal sample of 2048 rows."""
    X = cg.Standardizer().fit_transform(np.random.default_rng(4).normal(size=(2048, 5)))
    return cg.fit_floor_law(X, n_rep=12, seed=0)
