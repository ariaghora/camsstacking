"""
This module contains the class for CAMSStacker, a model stacking algorithm that
makes use of neural network to estimate the importance of each base model.

"""
from typing import List, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


class _WeightEstimator(BaseEstimator):
    """Estimate each base estimator's weight for each sample

    The weight estimator is implemented as a neural network with two hidden layers.
    The output layer uses a sigmoid activation function. The network is trained
    by minimizing binary cross-entropy loss, since the problem is formulated as
    a multioutput binary classification.

    """

    def __init__(
        self,
        hidden_layer_size: int,
        batch_size: Optional[int],
        device: str,
        verbose: bool = True,
    ) -> None:
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.scaler: StandardScaler = None
        self.net: _WeightEstimator = None
        self.losses: List[float] = None

    def build_network(self, input_neurons: int, output_neurons: int) -> torch.nn.Module:
        """
        The network is a simple feed-forward neural network with two hidden layers.
        TODO: provide flexibility to the user to choose the number of hidden layers,
              even the whole network architecture.
        """
        net = torch.nn.Sequential(
            torch.nn.Linear(input_neurons, self.hidden_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_layer_size, output_neurons),
            torch.nn.Sigmoid(),
        ).to(self.device)
        return net

    def fit(self, X: np.ndarray, y_weights: np.ndarray) -> "_WeightEstimator":
        """
        Fit the weight estimator on the training data
        """

        # We scale the input data to have zero mean and unit variance.
        # Usually this leads to a better convergence.
        self.scaler = StandardScaler().fit(X)
        X_nn = self.scaler.transform(X)

        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the Sequential model of the neural network.
        self.net = self.build_network(X_nn.shape[1], y_weights.shape[1])
        optim = torch.optim.Adam(self.net.parameters())
        loss_fn = torch.nn.BCELoss().to(self.device)

        # We specify the batch_size for the user when they did not specify it.
        if self.batch_size is None:
            self.batch_size = min(200, X_nn.shape[0])

        self.losses = []
        for epoch in tqdm(range(500), disable=(not self.verbose)):
            losses_epoch = []
            for iter in range(0, X_nn.shape[0], self.batch_size):
                optim.zero_grad()
                y_hat = self.net(
                    torch.from_numpy(X_nn[iter : iter + self.batch_size])
                    .float()
                    .to(self.device)
                )
                loss = loss_fn(
                    y_hat,
                    torch.from_numpy(y_weights[iter : iter + self.batch_size])
                    .float()
                    .to(self.device),
                )
                loss.backward()
                optim.step()

                losses_epoch.append(loss.item())
            self.losses.append(np.mean(losses_epoch))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the weight of each base estimator for each sample
        """
        X_transformed = self.scaler.transform(X)

        self.net.eval()
        with torch.no_grad():
            y_weights_hat = (
                self.net(torch.from_numpy(X_transformed).float().to(self.device))
                .cpu()
                .numpy()
            )

        return y_weights_hat


class CAMSStacker(BaseEstimator):
    """
    CAMSStacker is a model stacking algorithm that combines the outputs from
    multiple base estimators. A neural network is trained to estimate the weight
    of each base estimator for each sample. The weights are then used to
    determine the importance of each base estimator's output.

    Parameters
    ----------
    base_estimators : List[Any]
        The list of estimators that implements the `fit` and `predict_proba` methods.
    calibration_method : str, optional (default="isotonic")
        The calibration method to use for estimator calibration. Possible values are
        "isotonic" and "sigmoid".
    cv : int, optional (default=5)
        The number of folds to use for cross-validation in calibration.
    hidden_layer_size : int, optional (default=50)
        The number of neurons in the hidden layer. All hidden layers are fully
        connected and have the same number of neurons.
    batch_size : int, optional (default=None)
        The batch size for the training. If None, the batch size is set to
        the min(200, n_samples).
    verbose : bool, optional
        If True, the progress of the training is shown.

    """

    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        calibration_method: str = "isotonic",
        cv_calibration: int = 5,
        cv_stacking: int = 5,
        refit: bool = True,
        batch_size: Optional[int] = None,
        hidden_layer_size: int = 100,
        device: Optional[str] = None,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.base_estimators = base_estimators
        self.calibration_method = calibration_method
        self.cv_calibration = cv_calibration
        self.cv_stacking = cv_stacking
        self.refit = refit
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.weight_estimator: _WeightEstimator = None
        self.encoder: OneHotEncoder = None
        self.cal_estimators: BaseEstimator = None
        self.is_fitted_ = False

    def calculate_estimator_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the weight of each base estimator for each sample
        """
        result = np.hstack(
            [
                (e.predict(X) == y).astype(float).reshape(-1, 1)
                for e in self.cal_estimators
            ]
        )

        # Supress warnings about the zero division. It is expected that sometimes
        # for some samples, base estimators are not able to predict the target
        # variable. We will replace NaN values with some constant.
        with np.errstate(divide="ignore", invalid="ignore"):
            result = result / result.sum(1, keepdims=True)
        nan_replacement = 1 / result.shape[1]
        result = np.nan_to_num(result, nan=nan_replacement)
        return result

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CAMSStacker":
        """
        Fit the estimator on the training data
        """
        self.encoder = OneHotEncoder().fit(y.reshape(-1, 1))

        estimator_weights = []
        for idx_train, idx_valid in StratifiedKFold(
            n_splits=self.cv_stacking, shuffle=False
        ).split(X, y):
            X_train, X_valid = X[idx_train], X[idx_valid]
            y_train, y_valid = y[idx_train], y[idx_valid]

            # Some estimators such as decision tree and SVM usually output overconfident
            # probabilities. We use the calibration method to reduce such an issue.
            # In this case, we parallelly fit the calibrated estimators using sklearn's
            # CalibratedClassifierCV using cross-validation.
            self.cal_estimators = Parallel(n_jobs=self.n_jobs)(
                delayed(
                    CalibratedClassifierCV(
                        e, method=self.calibration_method, cv=self.cv_calibration
                    ).fit
                )(X_train, y_train)
                for e in self.base_estimators
            )

            # Collect the "ground-truth" weights of each base estimator for current fold
            estimator_weight_per_fold = self.calculate_estimator_weights(
                X_valid, y_valid
            )
            estimator_weights.append(estimator_weight_per_fold)

        # Stack the estimator weights from each fold as a np.ndarray.
        # This will be the target variable for the neural network.
        y_weights = np.vstack(estimator_weights)

        # Train the neural net for weight estimation
        self.weight_estimator = _WeightEstimator(
            hidden_layer_size=self.hidden_layer_size,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        ).fit(X, y_weights)

        # Finally, re-fit (calibrated) base estimators with full-dataset if
        # refit is True.
        if self.refit:
            self.cal_estimators = Parallel(n_jobs=self.n_jobs)(
                delayed(
                    CalibratedClassifierCV(
                        e, method=self.calibration_method, cv=self.cv_calibration
                    ).fit
                )(X, y)
                for e in self.base_estimators
            )

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each class for each sample
        """
        # Check if the model is fitted
        if not self.is_fitted_:
            raise NotFittedError("The model is not fitted yet.")

        # Calculate the weight of each base estimator for each sample.
        y_weights_hat = self.weight_estimator.predict(X)

        weighted_votes = sum(
            [
                np.multiply(e.predict_proba(X), y_weights_hat[:, [i]])
                for i, e in enumerate(self.cal_estimators)
            ]
        )

        # To interpret the output as probabilities, we need to normalize
        # the weighted votes.
        normalized_weighted_votes = weighted_votes / weighted_votes.sum(
            1, keepdims=True
        )
        return normalized_weighted_votes

    def predict(self, X):
        """
        Predict the class for each sample
        """
        weighted_votes = self.predict_proba(X)

        # Finally return the discrete class predictions
        labels = self.encoder.inverse_transform(weighted_votes).squeeze()
        return labels

    def score(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))
