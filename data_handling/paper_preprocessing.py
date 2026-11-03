"""Training-fitted preprocessing for a paper-style gastric cancer dataset.

The reference paper does not publish its exact outlier cutoffs or K-means
cluster count. This implementation uses the project's documented extreme
entry-error rules and makes ``n_clusters`` explicit (default 5). Any learned
imputation state is fitted on the training partition and reused on test data.
"""

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PAPER_INPUT_FEATURES = [
    "age", "gender", "neu", "lym", "neu_lym", "hb", "rdw", "plt", "alb",
    "alt", "tb", "cr", "tg", "hdl", "ldl", "lpa", "cea", "ca125", "ca199", "ca724",
]

PAPER_SELECTED_FEATURES = [
    "gender", "age", "neu_lym", "hb", "alb", "cea", "ca125", "ca199",
]


class PaperFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Impute documented extreme values with train-fitted K-means centroids.

    Input must be a DataFrame containing the paper's 20 initial predictors.
    Output is a DataFrame containing the eight predictors reported as
    independent characteristics by the paper. No target column is accepted as
    an input feature.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    @staticmethod
    def _mark_known_extreme_values(data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data[PAPER_INPUT_FEATURES].copy().apply(pd.to_numeric, errors="coerce")
        # Known severe unit/decimal-entry anomalies documented for this cohort.
        for column, threshold in (("hb", 300.0), ("alb", 100.0), ("rdw", 40.0)):
            cleaned.loc[cleaned[column] > threshold, column] = np.nan
        return cleaned

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        self._validate_columns(X)
        cleaned = self._mark_known_extreme_values(X)
        complete_rows = cleaned.dropna(axis=0, how="any")
        if len(complete_rows) < 2:
            raise ValueError("At least two complete training rows are required for K-means imputation.")

        self.feature_names_in_ = np.asarray(PAPER_INPUT_FEATURES, dtype=object)
        self.n_clusters_ = min(int(self.n_clusters), len(complete_rows))
        if self.n_clusters_ < 1:
            raise ValueError("n_clusters must be positive.")

        self.scaler_ = StandardScaler().fit(complete_rows)
        scaled_complete = self.scaler_.transform(complete_rows)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters_,
            random_state=self.random_state,
            n_init=10,
        ).fit(scaled_complete)
        self.centroids_original_scale_ = self.scaler_.inverse_transform(self.kmeans_.cluster_centers_)
        self.n_training_rows_ = int(len(X))
        self.n_complete_training_rows_ = int(len(complete_rows))
        self.training_anomalies_ = {
            column: int(cleaned[column].isna().sum())
            for column in ("hb", "alb", "rdw")
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "kmeans_"):
            raise RuntimeError("Call fit on the training data before transform.")
        self._validate_columns(X)
        cleaned = self._mark_known_extreme_values(X)

        for row_index in cleaned.index[cleaned.isna().any(axis=1)]:
            row = cleaned.loc[row_index]
            observed_mask = row.notna().to_numpy()
            missing_mask = ~observed_mask
            if not missing_mask.any():
                continue
            if not observed_mask.any():
                raise ValueError(f"Cannot K-means-impute row {row_index}: all paper predictors are missing.")

            observed_scaled = (
                row.to_numpy(dtype=float)[observed_mask] - self.scaler_.mean_[observed_mask]
            ) / self.scaler_.scale_[observed_mask]
            centroid_observed = self.kmeans_.cluster_centers_[:, observed_mask]
            distances = np.sum((centroid_observed - observed_scaled) ** 2, axis=1)
            nearest_cluster = int(np.argmin(distances))
            row_values = row.to_numpy(dtype=float)
            row_values[missing_mask] = self.centroids_original_scale_[nearest_cluster, missing_mask]
            cleaned.loc[row_index] = row_values

        if cleaned[PAPER_SELECTED_FEATURES].isna().any().any():
            raise ValueError("Selected predictors still contain missing values after imputation.")
        return cleaned[PAPER_SELECTED_FEATURES].copy()

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PaperFeaturePreprocessor expects a pandas DataFrame.")
        missing = [column for column in PAPER_INPUT_FEATURES if column not in X.columns]
        if missing:
            raise ValueError(f"Required paper predictors are missing: {missing}")
