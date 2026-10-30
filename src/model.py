"""Baseline genre classifier with standardized features and probabilities."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_classifier(random_state: int = 42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=random_state),
    )
