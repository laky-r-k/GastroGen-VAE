import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler

def clean_anomalies_shared(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df_clean = df.copy()
    hb_mask = df_clean["hb"] > 300.0
    if hb_mask.sum() > 0:
        df_clean.loc[hb_mask, "hb"] = df_clean.loc[hb_mask, "hb"] / 10.0
        if verbose: print(f"[Preprocessing] Corrected {hb_mask.sum()} extreme 'hb' anomaly values (divided by 10).")

    alb_mask = df_clean["alb"] > 100.0
    if alb_mask.sum() > 0:
        df_clean.loc[alb_mask, "alb"] = df_clean.loc[alb_mask, "alb"] / 10.0
        if verbose: print(f"[Preprocessing] Corrected {alb_mask.sum()} extreme 'alb' anomaly values (divided by 10).")

    rdw_valid = df_clean[df_clean["rdw"] <= 40.0]["rdw"]
    if len(rdw_valid) > 0:
        rdw_cap = rdw_valid.quantile(0.99)
        rdw_mask = df_clean["rdw"] > 40.0
        if rdw_mask.sum() > 0:
            df_clean.loc[rdw_mask, "rdw"] = rdw_cap
            if verbose: print(f"[Preprocessing] Clipped {rdw_mask.sum()} extreme 'rdw' outlier values to 99th percentile ({rdw_cap:.2f}%).")
    return df_clean

def engineer_features_shared(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    skewed_cols = ["cea", "ca199", "ca125", "ca724", "lpa", "alt", "tb", "tg", "neu_lym"]
    for col in skewed_cols:
        if col in df_eng.columns:
            df_eng[f"log_{col}"] = np.log1p(df_eng[col].clip(lower=0))

    df_eng["nlr_computed"] = df_eng["neu"] / (df_eng["lym"] + 1e-5)
    df_eng["hb_rdw_ratio"] = df_eng["hb"] / (df_eng["rdw"] + 1e-5)
    df_eng["plt_lym_ratio"] = df_eng["plt"] / (df_eng["lym"] + 1e-5)
    
    tumor_markers = ["log_cea", "log_ca199", "log_ca724", "log_ca125"]
    if all(m in df_eng.columns for m in tumor_markers):
        df_eng["tumor_marker_composite"] = sum(df_eng[m] for m in tumor_markers)
        
    df_eng["atherogenic_index"] = df_eng["chol"] / (df_eng["hdl"] + 1e-5)
    df_eng["alb_tb_ratio"] = df_eng["alb"] / (df_eng["tb"] + 1e-5)
    return df_eng

class PreprocessingPipeline:
    def __init__(
        self,
        feature_mode: str = "optimized_subset",
        scaler_type: str = "robust",
        apply_scaling: bool = False,
    ):
        self.feature_mode = feature_mode
        self.scaler_type = scaler_type
        self.apply_scaling = apply_scaling
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.selected_feature_names: List[str] = []

    def clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        return clean_anomalies_shared(df, verbose=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return engineer_features_shared(df)

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y = df["label"] if "label" in df.columns else None
        if self.feature_mode == "raw_cleaned":
            feature_cols = [c for c in df.columns if c not in ["patient_id", "label"] and not c.startswith("log_")]
        elif self.feature_mode == "optimized_subset":
            feature_cols = [
                "gender", "age", "log_neu_lym", "nlr_computed", "hb", "hct", "rdw",
                "hb_rdw_ratio", "plt_lym_ratio", "alb", "log_alt", "log_tb",
                "alb_tb_ratio", "cr", "log_tg", "chol", "hdl", "atherogenic_index",
                "log_lpa", "log_cea", "log_ca199", "log_ca125", "log_ca724", "tumor_marker_composite",
            ]
            feature_cols = [c for c in feature_cols if c in df.columns]
        else:
            feature_cols = [c for c in df.columns if c not in ["patient_id", "label"]]

        self.selected_feature_names = feature_cols
        X = df[feature_cols]
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)
        X_df, y_series = self.select_features(df_eng)
        X = X_df.values
        y = y_series.values if y_series is not None else None
        if self.apply_scaling:
            X = self.scaler.fit_transform(X)
        return X, y, self.selected_feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)
        X_df = df_eng[self.selected_feature_names]
        y_series = df_eng["label"] if "label" in df_eng.columns else None
        X = X_df.values
        if self.apply_scaling:
            X = self.scaler.transform(X)
        y = y_series.values if y_series is not None else None
        return X, y

class VAEPreprocessingPipeline:
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        return clean_anomalies_shared(df, verbose=False)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return engineer_features_shared(df)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)
        feature_cols = [c for c in df_eng.columns if c not in ["patient_id", "label"]]
        self.feature_names = feature_cols
        X_raw = df_eng[feature_cols].values
        y = df_eng["label"].values if "label" in df_eng.columns else None
        X_scaled = self.scaler.fit_transform(X_raw)
        self.is_fitted = True
        return X_scaled, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        df_clean = self.clean_anomalies(df)
        df_eng = self.engineer_features(df_clean)
        X_raw = df_eng[self.feature_names].values
        y = df_eng["label"].values if "label" in df_eng.columns else None
        X_scaled = self.scaler.transform(X_raw)
        return X_scaled, y

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)
