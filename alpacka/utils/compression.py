"""Compression utils."""

import joblib


def dump(value, path):
    joblib.dump(
        value,
        path,

        protocol=4,

        compress=('gzip', 4),
    )


def load(path):
    return joblib.load(path)
