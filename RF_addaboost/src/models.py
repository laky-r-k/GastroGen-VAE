"""
Model Architectures and Factory Module for Random Forest, AdaBoost, and Hybrid Ensembles.
"""

import os
import joblib
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def build_random_forest(
    n_estimators: int = 250,
    max_depth: Optional[int] = 6,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Constructs an optimized Random Forest Classifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def build_adaboost(
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    random_state: int = 42,
) -> AdaBoostClassifier:
    """
    Constructs an optimized AdaBoost Classifier with shallow decision stumps.
    """
    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
    )
    return AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def build_soft_voting_ensemble(
    rf_model: Optional[RandomForestClassifier] = None,
    ada_model: Optional[AdaBoostClassifier] = None,
    weights: Optional[list] = None,
) -> VotingClassifier:
    """
    Constructs a Soft Voting Ensemble combining Random Forest and AdaBoost.
    Averages predicted class probabilities for superior calibration and variance reduction.
    """
    if rf_model is None:
        rf_model = build_random_forest()
    if ada_model is None:
        ada_model = build_adaboost()

    return VotingClassifier(
        estimators=[("random_forest", rf_model), ("adaboost", ada_model)],
        voting="soft",
        weights=weights or [1.0, 1.0],
        n_jobs=-1,
    )


def build_stacking_ensemble(
    rf_model: Optional[RandomForestClassifier] = None,
    ada_model: Optional[AdaBoostClassifier] = None,
    final_estimator: Optional[Any] = None,
) -> StackingClassifier:
    """
    Constructs a Stacking Classifier with RF and AdaBoost base estimators
    and a regularized Logistic Regression meta-learner.
    """
    if rf_model is None:
        rf_model = build_random_forest()
    if ada_model is None:
        ada_model = build_adaboost()
    if final_estimator is None:
        final_estimator = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=500)

    return StackingClassifier(
        estimators=[("random_forest", rf_model), ("adaboost", ada_model)],
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
    )


def save_model(model: Any, filepath: str) -> None:
    """
    Serializes and saves a trained scikit-learn model to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[ModelRegistry] Successfully saved model to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Loads a serialized model from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    model = joblib.load(filepath)
    print(f"[ModelRegistry] Successfully loaded model from: {filepath}")
    return model
