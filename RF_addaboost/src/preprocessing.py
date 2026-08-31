"""
Data Preprocessing, Cleaning, and Clinical Feature Engineering Module.
Incorporates data-driven corrections and clinical domain knowledge from the EDA.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler


class PreprocessingPipeline:
    """
    End-to-end preprocessing, cleaning, and feature engineering pipeline.
    """

    def __init__(
        self,
        feature_mode: str = "optimized_subset",
        scaler_type: str = "robust",
        apply_scaling: bool = False,
    ):
        """
        Args:
            feature_mode: 'optimized_subset', 'all_engineered', or 'raw_cleaned'
            scaler_type: 'robust' (RobustScaler) or 'standard' (StandardScaler)
            apply_scaling: Whether to scale features (trees don't strictly require it,
                           but helps meta-learners in stacking ensembles).
        """
        self.feature_mode = feature_mode
        self.scaler_type = scaler_type
        self.apply_scaling = apply_scaling
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.selected_feature_names: List[str] = []

    def clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects physiological typographical anomalies identified during EDA:
          1. hb == 1222.0 -> divided by 10 (122.2 g/L)
          2. alb == 397.0 -> divided by 10 (39.7 g/L)
          3. rdw > 40 -> clipped to 99th percentile of valid clinical range
        """
        df_clean = df.copy()

        # Fix Hemoglobin decimal shift error (> 300 g/L is non-viable)
        hb_mask = df_clean["hb"] > 300.0
        if hb_mask.sum() > 0:
            df_clean.loc[hb_mask, "hb"] = df_clean.loc[hb_mask, "hb"] / 10.0
            print(f"[Preprocessing] Corrected {hb_mask.sum()} extreme 'hb' anomaly values (divided by 10).")

        # Fix Albumin decimal shift error (> 100 g/L is non-viable)
        alb_mask = df_clean["alb"] > 100.0
        if alb_mask.sum() > 0:
            df_clean.loc[alb_mask, "alb"] = df_clean.loc[alb_mask, "alb"] / 10.0
            print(f"[Preprocessing] Corrected {alb_mask.sum()} extreme 'alb' anomaly values (divided by 10).")

        # Fix RDW typographical outliers (> 40%)
        rdw_valid = df_clean[df_clean["rdw"] <= 40.0]["rdw"]
        rdw_cap = rdw_valid.quantile(0.99)
        rdw_mask = df_clean["rdw"] > 40.0
        if rdw_mask.sum() > 0:
            df_clean.loc[rdw_mask, "rdw"] = rdw_cap
            print(f"[Preprocessing] Clipped {rdw_mask.sum()} extreme 'rdw' outlier values to 99th percentile ({rdw_cap:.2f}%).")

        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies non-linear transformations to heavily skewed markers
        and derives clinically validated diagnostic indices.
        """
        df_eng = df.copy()

        # 1. Log-transform heavy-tailed serum tumor markers and enzymes
        skewed_cols = ["cea", "ca199", "ca125", "ca724", "lpa", "alt", "tb", "tg", "neu_lym"]
        for col in skewed_cols:
            if col in df_eng.columns:
                df_eng[f"log_{col}"] = np.log1p(df_eng[col].clip(lower=0))

        # 2. Complete Blood Count & Inflammatory Indices
        df_eng["nlr_computed"] = df_eng["neu"] / (df_eng["lym"] + 1e-5)
        df_eng["hb_rdw_ratio"] = df_eng["hb"] / (df_eng["rdw"] + 1e-5)  # Hemoglobin-to-RDW ratio (Prognostic marker)
        df_eng["plt_lym_ratio"] = df_eng["plt"] / (df_eng["lym"] + 1e-5)  # Platelet-to-Lymphocyte Ratio (PLR)

        # 3. Tumor Marker Composite Score (Combined oncological burden)
        df_eng["tumor_marker_composite"] = (
            df_eng["log_cea"] + df_eng["log_ca199"] + df_eng["log_ca724"] + df_eng["log_ca125"]
        )

        # 4. Metabolic & Liver Function Indices
        df_eng["atherogenic_index"] = df_eng["chol"] / (df_eng["hdl"] + 1e-5)  # Total Chol / HDL
        df_eng["alb_tb_ratio"] = df_eng["alb"] / (df_eng["tb"] + 1e-5)  # Albumin-to-Bilirubin proxy

        return df_eng

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Filters and selects features based on the chosen mode.
        """
        y = df["label"]

        if self.feature_mode == "raw_cleaned":
            feature_cols = [c for c in df.columns if c not in ["patient_id", "label"] and not c.startswith("log_")]
        elif self.feature_mode == "optimized_subset":
            # Select key non-redundant predictors identified during EDA
            feature_cols = [
                "gender",
                "age",
                "log_neu_lym",
                "nlr_computed",
                "hb",
                "hct",
                "rdw",
                "hb_rdw_ratio",
                "plt_lym_ratio",
                "alb",
                "log_alt",
                "log_tb",
                "alb_tb_ratio",
                "cr",
                "log_tg",
                "chol",
                "hdl",
                "atherogenic_index",
                "log_lpa",
                "log_cea",
                "log_ca199",
                "log_ca125",
                "log_ca724",
                "tumor_marker_composite",
            ]
            # Ensure only columns present in df are included
            feature_cols = [c for c in feature_cols if c in df.columns]
        else:  # 'all_engineered'
            feature_cols = [c for c in df.columns if c not in ["patient_id", "label"]]

        self.selected_feature_names = feature_cols
        X = df[feature_cols]

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Full pipeline: Clean anomalies -> Engineer features -> Select features -> (Optional) Scale.
        """
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)
        X_df, y_series = self.select_features(df_eng)

        X = X_df.values
        y = y_series.values

        if self.apply_scaling:
            X = self.scaler.fit_transform(X)

        return X, y, self.selected_feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new unseen test data using fitted scaler and selected features.
        """
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)
        X_df = df_eng[self.selected_feature_names]
        y_series = df_eng["label"] if "label" in df_eng.columns else None

        X = X_df.values
        if self.apply_scaling:
            X = self.scaler.transform(X)

        y = y_series.values if y_series is not None else None
        return X, y
