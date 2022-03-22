"""
This script runs the classification experiment on all datasets.
It will output the results to a csv files and figures."""
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../")


import warnings

import numpy as np
import pandas as pd
import torch
from cams import CAMSStacker
from experiment.experiment_data_loader import available_datasets, get_dataset
from tqdm import tqdm

base_estimators = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    GaussianNB(),
]

cams_stacker_params = {
    "calibration_method": "sigmoid",
    "base_estimators": base_estimators,
    "hidden_layer_size": 200,
    "n_iter": 3000,
    "n_jobs": 1,
    "verbose": False,
}

linear_stack_params = {"estimators": [(str(e), e) for e in base_estimators]}


def run_training(datasets: Optional[List[str]] = None, random_state: int = 1) -> None:
    if datasets is None:
        datasets = available_datasets

    training_result = pd.DataFrame(index=datasets)

    os.makedirs("output", exist_ok=True)

    for dataset_name in tqdm(datasets):
        # show dataset name in the progress bar
        tqdm.write(f"Dataset: {dataset_name}")

        X_train, X_test, y_train, y_test = get_dataset(
            dataset_name, random_state=random_state
        )

        cams_stacker = CAMSStacker(**cams_stacker_params).fit(X_train, y_train)
        linear_stacker = StackingClassifier(**linear_stack_params).fit(X_train, y_train)

        score_cams_stacker = cams_stacker.score(X_test, y_test)
        score_linear_stacker = linear_stacker.score(X_test, y_test)

        training_result.loc[dataset_name, "cams_stacker"] = score_cams_stacker
        training_result.loc[dataset_name, "linear_stacker"] = score_linear_stacker

        # Save loss history
        plt.plot(cams_stacker.weight_estimator.losses)
        plt.xlabel("Iteration")
        plt.savefig(f"output/{dataset_name}_loss.png")
        plt.close()

    training_result.to_csv("output/training_result.csv")
    print("Training result saved to output/training_result.csv")
    print(training_result)


if __name__ == "__main__":
    # Supress all warnings
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to run the experiment on. If not specified, run on all datasets.",
    )

    warnings.filterwarnings("ignore")
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

    # Seed everything
    np.random.seed(1)
    torch.manual_seed(1)

    chosen_datasets = (
        parser.parse_args().dataset.split(",") if parser.parse_args().dataset else None
    )
    run_training(datasets=chosen_datasets)
