"""
   gradient boosting model definition
"""

from typing import Optional
from sklearn.ensemble import GradientBoostingClassifier

def build_gradient_boosting(
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: Optional[int] = 4,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Constructs an optimized Gradient Boosting Classifier.
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    )
