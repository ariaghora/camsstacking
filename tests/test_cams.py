import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cams import CAMSStacker
from typing import Tuple
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=1000,
        n_features=40,
        n_informative=15,
        n_redundant=5,
        random_state=1,
        n_classes=3,
    )
    return X, y


def test_fit() -> None:
    X, y = get_dataset()
    clf = CAMSStacker(
        base_estimators=[LogisticRegression(), DecisionTreeClassifier()]
    ).fit(X, y)
    assert clf.is_fitted_


def test_predict() -> None:
    X, y = get_dataset()
    clf = CAMSStacker(
        base_estimators=[LogisticRegression(), DecisionTreeClassifier()]
    ).fit(X, y)

    assert clf.predict(X).shape == (X.shape[0],)


def test_predict_proba() -> None:
    X, y = get_dataset()
    clf = CAMSStacker(
        base_estimators=[LogisticRegression(), DecisionTreeClassifier()]
    ).fit(X, y)

    assert clf.predict_proba(X).shape == (X.shape[0], 3)
