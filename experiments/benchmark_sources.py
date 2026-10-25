"""Provenance of the benchmark tables (Section 8.1).

For every source table of experiments/_benchmarks.py, the OpenML data id,
name and version, the MD5 checksum that OpenML publishes for the file, the
licence, and the shape of the table as downloaded (covertype comes from the
UCI copy distributed through scikit-learn). For every cell, the shape after
encoding and the SHA-256 digest of the encoded matrix (float64,
little-endian, row-major), so that a later download can be checked against
the data analysed here.

    python experiments/benchmark_sources.py

Downloads the tables from OpenML on first use. Writes
results/benchmark_sources.json (about a minute once the tables are cached).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

import _artifacts as art
from _benchmarks import CELLS, EXCLUDED, OPENML, Tables


def digest(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes()).hexdigest()


def source_record(source: str, frame) -> dict:
    rec = {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])}
    if source in OPENML:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(**OPENML[source], as_frame=True, parser="auto").details
        rec.update({"repository": "OpenML", "data_id": int(d["id"]),
                    "name": d["name"], "version": int(d["version"]),
                    "md5_checksum": d.get("md5_checksum"),
                    "licence": d.get("licence"),
                    "original_data_url": d.get("original_data_url")})
    else:
        rec.update({"repository": "UCI Machine Learning Repository, through "
                                  "sklearn.datasets.fetch_covtype"})
    return rec


def main():
    t0 = time.time()
    tables = Tables()
    sources, cells = {}, []
    for role, group in (("analysed", CELLS), ("excluded", EXCLUDED)):
        for tag, source, enc in group:
            frame = tables.frame(source)
            if source not in sources:
                sources[source] = source_record(source, frame)
            X = tables.encoded(source, enc)
            cells.append({"cell": tag, "role": role, "source": source,
                          "encoding": enc, "rows": int(X.shape[0]),
                          "columns": int(X.shape[1]), "sha256": digest(X)})
            print(f"{tag:<20} {X.shape[0]:>8} x {X.shape[1]:<3} {cells[-1]['sha256'][:16]}",
                  flush=True)
    art.save("benchmark_sources.json", {
        "_definition": ("source tables of the benchmark study and digests of "
                        "the encoded matrices before standardisation and "
                        "subsampling"),
        "sources": sources, "cells": cells,
        "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
