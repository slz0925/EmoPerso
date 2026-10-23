"""Validated loading of cumulative precomputed feature CSVs."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import class_id_to_bits


class DataContractError(ValueError):
    """Input or source data does not satisfy its declared contract."""


@dataclass(frozen=True)
class FeatureSplit:
    features: np.ndarray
    labels: np.ndarray
    class_ids: np.ndarray
    # Zero-based, half-open data-row interval in cumulative eval.csv, not user IDs.
    source_rows: tuple[int, int]


@dataclass(frozen=True)
class FeatureSplits:
    train: FeatureSplit
    validation: FeatureSplit
    test: FeatureSplit


def _csv_rows(path, expected_header):
    """Yield logical CSV rows with useful path/physical-line diagnostics."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header != expected_header:
                if header is None:
                    raise DataContractError(f"{path}: missing header (empty file)")
                if len(header) != len(expected_header):
                    raise DataContractError(f"{path}: expected header with {len(expected_header)} "
                                            f"columns, got {len(header)}")
                mismatch = next(i for i, (actual, expected) in enumerate(zip(header, expected_header))
                                if actual != expected)
                raise DataContractError(f"{path}: header column {mismatch + 1}: expected "
                                        f"{expected_header[mismatch]!r}, got {header[mismatch]!r}")
            for row in reader:
                if len(row) != len(expected_header):
                    raise DataContractError(f"{path}: line {reader.line_num}: expected "
                                            f"{len(expected_header)} columns, got {len(row)}")
                yield reader.line_num, row
    except FileNotFoundError as exc:
        raise DataContractError(f"Missing required CSV: {path}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataContractError(f"Cannot read CSV {path}: {exc}") from exc


def _read_labels(path):
    rows = []
    for line, row in _csv_rows(path, [f"label{i}" for i in range(5)]):
        try:
            values = np.asarray(row, dtype=np.float64)
        except ValueError as exc:
            raise DataContractError(f"{path}: line {line}: labels must be numeric") from exc
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise DataContractError(f"{path}: line {line}: labels must be finite integers")
        if not np.all((values[:4] == 0) | (values[:4] == 1)) or not 0 <= values[4] <= 15:
            raise DataContractError(f"{path}: line {line}: expected binary label0..3 and label4 in [0,15]")
        if tuple(values[:4]) != class_id_to_bits(int(values[4])):
            raise DataContractError(f"{path}: line {line}: label4 is inconsistent with label0..3")
        rows.append(values.astype(np.int64))
    if not rows:
        raise DataContractError(f"{path}: label partition must not be empty")
    return np.asarray(rows, dtype=np.int64)


def _feature_rows(path, feature_dim, expected_count):
    count = 0
    for line, row in _csv_rows(path, [str(i) for i in range(feature_dim)]):
        if count >= expected_count:
            raise DataContractError(f"{path}: expected {expected_count} feature rows; found extra row at line {line}")
        try:
            values = np.asarray(row, dtype=np.float64)
        except ValueError as exc:
            raise DataContractError(f"{path}: line {line}: features must be numeric") from exc
        if not np.all(np.isfinite(values)):
            raise DataContractError(f"{path}: line {line}: features must be finite")
        yield count, line, values
        count += 1
    if count != expected_count:
        raise DataContractError(f"{path}: expected {expected_count} feature rows, got {count}")


def _feature_digest(values):
    # Numerical equality identifies both signs of zero. Work on a copy so the
    # canonical eval matrix and even caller-owned read-only arrays are untouched.
    canonical = values.copy()
    canonical[canonical == 0] = 0.0
    return hashlib.sha256(canonical.tobytes()).digest()


def load_feature_splits(directory, *, feature_dim=4096, dtype=np.float32) -> FeatureSplits:
    """Read each of the six required files once, without a full CSV/DataFrame copy.

    Storage order is train, test, eval(validation). eval.csv is the complete
    canonical stream; train.csv and test.csv must numerically match its prefixes.
    Prefix hashes use float64 values before output casting, so float32 rounding
    cannot hide disagreements. All returned partitions own separate feature
    storage, with read-only feature/label arrays. No file is written.
    """
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, (int, np.integer)) or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    try:
        dtype = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError("dtype must be float32 or float64") from exc
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    directory = Path(directory)
    names = ("train", "test", "eval")
    labels = [_read_labels(directory / f"{name}_label.csv") for name in names]
    counts = [len(array) for array in labels]
    stops = np.cumsum(counts).tolist()
    arrays = [np.empty((count, feature_dim), dtype=dtype) for count in counts]
    digests = []
    partition = 0
    path = directory / "eval.csv"
    for index, line, values in _feature_rows(path, feature_dim, stops[-1]):
        while index >= stops[partition]:
            partition += 1
        start = 0 if partition == 0 else stops[partition - 1]
        with np.errstate(over="ignore"):
            converted = values.astype(dtype)
        if not np.all(np.isfinite(converted)):
            raise DataContractError(f"{path}: line {line}: features overflow output dtype {dtype}")
        arrays[partition][index - start] = converted
        digests.append(_feature_digest(values))

    for name, count in (("train", stops[0]), ("test", stops[1])):
        path = directory / f"{name}.csv"
        for index, line, values in _feature_rows(path, feature_dim, count):
            if _feature_digest(values) != digests[index]:
                raise DataContractError(f"{path}: line {line}: cumulative prefix differs from eval.csv "
                                        f"at data row {index + 1}")

    splits = []
    start = 0
    for features, label_array, stop in zip(arrays, labels, stops):
        bits = label_array[:, :4].copy()
        class_ids = label_array[:, 4].copy()
        for array in (features, bits, class_ids):
            array.setflags(write=False)
        splits.append(FeatureSplit(features, bits, class_ids, (start, stop)))
        start = stop
    return FeatureSplits(train=splits[0], validation=splits[2], test=splits[1])
