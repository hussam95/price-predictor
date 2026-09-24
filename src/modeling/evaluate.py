"""Evaluation helpers kept separate from model fitting orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .common import FeatureSet, regression_metrics


def metric_record(
    *,
    category: str,
    model_name: str,
    feature_variant: str,
    evaluation_split: str,
    training_policy: str,
    y_true: pd.Series,
    prediction: np.ndarray,
) -> dict[str, object]:
    return {
        "category": category,
        "model": model_name,
        "feature_variant": feature_variant,
        "evaluation_split": evaluation_split,
        "training_policy": training_policy,
        **regression_metrics(y_true, prediction),
    }


def feature_permutation_importance(
    estimator: object,
    features: FeatureSet,
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    sample_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    if len(frame) > sample_rows:
        sampled = frame.sample(n=sample_rows, random_state=random_seed)
        sampled_target = target.loc[sampled.index]
    else:
        sampled = frame
        sampled_target = target
    result = permutation_importance(
        estimator,
        sampled[features.columns],
        sampled_target,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=random_seed,
        n_jobs=1,
    )
    return pd.DataFrame({
        "feature": features.columns,
        "importance_mean_mae_increase": result.importances_mean,
        "importance_std": result.importances_std,
        "sample_rows": len(sampled),
    }).sort_values("importance_mean_mae_increase", ascending=False)
