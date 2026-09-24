"""Shared modeling, evaluation, and reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_SEED = 42
SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True)
class FeatureSet:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Group categories using frequencies learned from fitting rows only."""

    def __init__(self, columns: tuple[str, ...], min_count: int = 20, label: str = "__RARE_OR_UNSEEN__"):
        self.columns = columns
        self.min_count = min_count
        self.label = label

    def fit(self, X: pd.DataFrame, y: object = None) -> "RareCategoryGrouper":
        self.frequent_values_ = {
            column: set(X[column].astype("string").value_counts().loc[lambda counts: counts >= self.min_count].index)
            for column in self.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for column in self.columns:
            values = result[column].astype("string")
            result[column] = values.where(values.isin(self.frequent_values_[column]), self.label)
        return result


class IndexedTargetRegressor(RegressorMixin, BaseEstimator):
    """Model a target in constant-index units, then restore nominal SGD."""

    def __init__(self, estimator: object, index_column: str = "lagged_rpi", base_index: float = 100.0):
        self.estimator = estimator
        self.index_column = index_column
        self.base_index = base_index

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "IndexedTargetRegressor":
        index = pd.to_numeric(X[self.index_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(index).all() or (index <= 0).any():
            raise ValueError("Indexed target requires complete positive index values")
        adjusted = np.asarray(y, dtype=float) * self.base_index / index
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, adjusted)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        index = pd.to_numeric(X[self.index_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(index).all() or (index <= 0).any():
            raise ValueError("Indexed target requires complete positive index values")
        return np.asarray(self.estimator_.predict(X), dtype=float) * index / self.base_index


def assert_split_isolation(frame: pd.DataFrame, expected: dict[str, int]) -> None:
    """Validate exact split counts and chronological ordering."""
    actual = frame["data_split"].value_counts().to_dict()
    for split, count in expected.items():
        if int(actual.get(split, 0)) != count:
            raise ValueError(f"{split}: expected {count:,} rows, found {actual.get(split, 0):,}")
    dates = pd.to_datetime(frame["transaction_date"], errors="raise")
    boundaries = {
        split: (dates[frame["data_split"].eq(split)].min(), dates[frame["data_split"].eq(split)].max())
        for split in SPLIT_ORDER
    }
    if not (boundaries["train"][1] < boundaries["validation"][0] <= boundaries["validation"][1] < boundaries["test"][0]):
        raise ValueError(f"Chronological split isolation failed: {boundaries}")


def assert_features_safe(features: FeatureSet, target: str, forbidden: Iterable[str]) -> None:
    selected = set(features.columns)
    prohibited = {target, *forbidden}
    overlap = selected & prohibited
    if overlap:
        raise ValueError(f"Leakage/reference fields selected: {sorted(overlap)}")
    if len(selected) != len(features.columns):
        raise ValueError("Duplicate model feature")


def make_linear_pipeline(
    features: FeatureSet,
    *,
    rare_columns: tuple[str, ...] = (),
    rare_min_count: int = 20,
) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, list(features.numeric)),
        ("categorical", categorical, list(features.categorical)),
    ])
    steps: list[tuple[str, object]] = []
    if rare_columns:
        steps.append(("rare_categories", RareCategoryGrouper(rare_columns, min_count=rare_min_count)))
    steps.extend([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=10.0, solver="lsqr")),
    ])
    return Pipeline(steps)


def make_hist_pipeline(
    features: FeatureSet,
    *,
    max_iter: int = 180,
    min_samples_leaf: int = 40,
) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, list(features.numeric)),
        ("categorical", categorical, list(features.categorical)),
    ])
    categorical_mask = [False] * len(features.numeric) + [True] * len(features.categorical)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.08,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def regression_metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Prediction shape {predicted.shape} does not match target {actual.shape}")
    median_target = float(np.median(actual))
    mae = float(mean_absolute_error(actual, predicted))
    return {
        "rows": int(len(actual)),
        "median_target": median_target,
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mae_percent_of_median": float(100.0 * mae / median_target),
    }


def absolute_error_percentiles(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    errors = np.abs(y_true.to_numpy(dtype=float) - np.asarray(prediction, dtype=float))
    return {
        "median_absolute_error": float(np.quantile(errors, 0.50)),
        "p80_absolute_error": float(np.quantile(errors, 0.80)),
        "p90_absolute_error": float(np.quantile(errors, 0.90)),
        "p95_absolute_error": float(np.quantile(errors, 0.95)),
    }


def evaluate_groups(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    group_column: str,
    *,
    minimum_rows: int,
) -> pd.DataFrame:
    work = pd.DataFrame({
        "group": frame[group_column].astype("string").fillna("MISSING"),
        "actual": frame["_target"].to_numpy(dtype=float),
        "prediction": np.asarray(prediction, dtype=float),
    })
    work["absolute_error"] = (work["actual"] - work["prediction"]).abs()
    work["signed_error"] = work["prediction"] - work["actual"]
    grouped = work.groupby("group", observed=True).agg(
        rows=("actual", "size"),
        median_target=("actual", "median"),
        mae=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
        mean_signed_error=("signed_error", "mean"),
    ).reset_index()
    grouped = grouped.loc[grouped["rows"].ge(minimum_rows)].copy()
    grouped["mae_percent_of_median"] = 100.0 * grouped["mae"] / grouped["median_target"]
    grouped.insert(0, "dimension", group_column)
    return grouped.sort_values(["mae", "rows"], ascending=[False, False])


def add_analysis_bands(frame: pd.DataFrame, train_target: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    q25, q50, q75 = train_target.quantile([0.25, 0.50, 0.75]).tolist()
    result["price_band"] = pd.cut(
        result["_target"],
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=["low", "lower_middle", "upper_middle", "high"],
        include_lowest=True,
    )
    if "nearest_mrt_distance_m" in result:
        result["mrt_distance_band"] = pd.cut(
            result["nearest_mrt_distance_m"],
            bins=[-np.inf, 500, 1000, 2000, np.inf],
            labels=["0-500m", "500-1000m", "1000-2000m", "2000m+"],
        ).astype("string").fillna("missing/ineligible")
    return result


def model_parameters(estimator: Pipeline) -> dict[str, Any]:
    params = estimator.named_steps["model"].get_params(deep=False)
    return {key: value for key, value in params.items() if isinstance(value, (str, int, float, bool, type(None)))}
