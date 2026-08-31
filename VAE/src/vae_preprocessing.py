"""
Preprocessing and Scaling Pipeline for Tabular Variational Autoencoder.
Handles anomaly sanitation, log-transforms, clinical feature engineering, and normalization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler


class VAEPreprocessingPipeline:
    """
    Preprocessing pipeline for Tabular VAE + downstream classifiers.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects entry typos and physiological anomalies identified during EDA.
        """
        df_clean = df.copy()

        # 1. Fix Hemoglobin typo (1222.0 -> 122.2)
        hb_mask = df_clean["hb"] > 300.0
        if hb_mask.sum() > 0:
            df_clean.loc[hb_mask, "hb"] = df_clean.loc[hb_mask, "hb"] / 10.0

        # 2. Fix Albumin typo (397.0 -> 39.7)
        alb_mask = df_clean["alb"] > 100.0
        if alb_mask.sum() > 0:
            df_clean.loc[alb_mask, "alb"] = df_clean.loc[alb_mask, "alb"] / 10.0

        # 3. Clip RDW typos (>40%) to 99th percentile
        rdw_valid = df_clean[df_clean["rdw"] <= 40.0]["rdw"]
        rdw_cap = rdw_valid.quantile(0.99)
        rdw_mask = df_clean["rdw"] > 40.0
        if rdw_mask.sum() > 0:
            df_clean.loc[rdw_mask, "rdw"] = rdw_cap

        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives non-linear log transformations and clinical indices.
        """
        df_eng = df.copy()

        # Log transform skewed tumor markers & enzymes
        skewed_cols = ["cea", "ca199", "ca125", "ca724", "lpa", "alt", "tb", "tg", "neu_lym"]
        for col in skewed_cols:
            if col in df_eng.columns:
                df_eng[f"log_{col}"] = np.log1p(df_eng[col].clip(lower=0))

        # Clinical composite ratios
        df_eng["nlr_computed"] = df_eng["neu"] / (df_eng["lym"] + 1e-5)
        df_eng["hb_rdw_ratio"] = df_eng["hb"] / (df_eng["rdw"] + 1e-5)
        df_eng["plt_lym_ratio"] = df_eng["plt"] / (df_eng["lym"] + 1e-5)
        df_eng["tumor_marker_composite"] = (
            df_eng["log_cea"] + df_eng["log_ca199"] + df_eng["log_ca724"] + df_eng["log_ca125"]
        )
        df_eng["atherogenic_index"] = df_eng["chol"] / (df_eng["hdl"] + 1e-5)
        df_eng["alb_tb_ratio"] = df_eng["alb"] / (df_eng["tb"] + 1e-5)

        return df_eng

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fits scaler and transforms dataframe into normalized numpy feature matrix.
        """
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)

        # Select all clinical and engineered features (excluding IDs and label)
        feature_cols = [c for c in df_eng.columns if c not in ["patient_id", "label"]]
        self.feature_names = feature_cols

        X_raw = df_eng[feature_cols].values
        y = df_eng["label"].values

        X_scaled = self.scaler.fit_transform(X_raw)
        self.is_fitted = True

        return X_scaled, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms new test dataframe using fitted scaler parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)

        X_raw = df_eng[self.feature_names].values
        y = df_eng["label"].values if "label" in df_eng.columns else None

        X_scaled = self.scaler.transform(X_raw)
        return X_scaled, y

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverts scaling back to original feature units.
        """
        return self.scaler.inverse_transform(X_scaled)
