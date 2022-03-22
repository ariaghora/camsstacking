"""
A helper function to load the datasets for experiments.
"""
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces, load_breast_cancer, fetch_openml


def load_faces_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_olivetti_faces(return_X_y=True)


def load_breast_cancer_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return load_breast_cancer(return_X_y=True)


def load_credit_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_openml(name="credit-g", return_X_y=True, as_frame=False)


def load_har_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_openml(name="har", return_X_y=True, as_frame=False)


def load_japanese_vowels_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_openml(name="JapaneseVowels", return_X_y=True, as_frame=False)


def load_vowels_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_openml(name="vowel", return_X_y=True, as_frame=False)


def load_phishing_wrapper() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_openml(name="PhishingWebsites", return_X_y=True, as_frame=False)


dataset_loader_dict = {
    "faces": load_faces_wrapper,
    "breast_cancer": load_breast_cancer_wrapper,
    "credit": load_credit_wrapper,
    "har": load_har_wrapper,
    "japanese_vowels": load_japanese_vowels_wrapper,
    "vowels": load_vowels_wrapper,
    "phishing": load_phishing_wrapper,
}

available_datasets = list(dataset_loader_dict.keys())


def get_dataset(
    dataset_name: str,
    random_state: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if dataset_name not in dataset_loader_dict:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    X, y = dataset_loader_dict[dataset_name]()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
