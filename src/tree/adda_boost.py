""" 
   adaboost model definition
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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