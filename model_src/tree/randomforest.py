"""
  randomforest model definition
"""


from typing import  Optional
from sklearn.ensemble import RandomForestClassifier



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









