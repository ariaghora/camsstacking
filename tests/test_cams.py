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

    # The output of predict_proba should be a 2D array of shape
    # (n_samples, n_classes)
    assert clf.predict_proba(X).shape == (X.shape[0], 3)

    # The sum of the probabilities should be 1, but due to numerical
    # errors it is not always the case. So we check that the sum is
    # close to 1 within a small margin.
    assert (clf.predict_proba(X).sum(axis=1).max() - 1) < 1e-10
