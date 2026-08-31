"""
Downstream Classifiers Module for VAE Compressed Latent Space and Hybrid Representations.
Trains Random Forest, AdaBoost, Soft Voting, and Stacking Classifiers on latent embeddings.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def build_rf_classifier(random_state: int = 42) -> RandomForestClassifier:
    """
    Constructs a Random Forest classifier optimized for latent spaces.
    """
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )


def build_adaboost_classifier(random_state: int = 42) -> AdaBoostClassifier:
    """
    Constructs an AdaBoost classifier with shallow decision trees.
    """
    base_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
    )
    return AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=150,
        learning_rate=0.05,
        random_state=random_state,
    )


def build_soft_voting(rf=None, ada=None, random_state: int = 42) -> VotingClassifier:
    """
    Constructs a Soft Voting Ensemble blending RF and AdaBoost probabilities.
    """
    rf = rf or build_rf_classifier(random_state=random_state)
    ada = ada or build_adaboost_classifier(random_state=random_state)
    return VotingClassifier(
        estimators=[("rf", rf), ("ada", ada)],
        voting="soft",
        weights=[1.0, 1.0],
        n_jobs=1,
    )


def build_stacking(rf=None, ada=None, random_state: int = 42) -> StackingClassifier:
    """
    Constructs a Stacking Classifier with Logistic Regression meta-learner.
    """
    rf = rf or build_rf_classifier(random_state=random_state)
    ada = ada or build_adaboost_classifier(random_state=random_state)
    meta = LogisticRegression(C=1.0, penalty="l2", random_state=random_state, max_iter=500)
    return StackingClassifier(
        estimators=[("rf", rf), ("ada", ada)],
        final_estimator=meta,
        cv=5,
        n_jobs=1,
    )


def get_model_suite(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of all candidate classification models.
    """
    return {
        "Random Forest": lambda: build_rf_classifier(random_state=random_state),
        "AdaBoost": lambda: build_adaboost_classifier(random_state=random_state),
        "Soft Voting (RF + Ada)": lambda: build_soft_voting(random_state=random_state),
        "Stacking (RF + Ada)": lambda: build_stacking(random_state=random_state),
    }


def save_joblib(obj: Any, filepath: str) -> None:
    """
    Helper to serialize python objects using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"[ModelRegistry] Saved artifact to: {filepath}")


def load_joblib(filepath: str) -> Any:
    """
    Helper to deserialize python objects from joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)
