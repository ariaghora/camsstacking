from typing import List, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


class WeightEstimator(BaseEstimator):
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
        self.weight_estimator: WeightEstimator = None

    def fit(self, X: np.ndarray, y_weights: np.ndarray) -> "WeightEstimator":
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
        # The network is a simple feed-forward neural network with two hidden layers.
        # TODO: provide flexibility to the user to choose the number of hidden layers,
        #       even the whole network architecture.
        self.weight_estimator = torch.nn.Sequential(
            torch.nn.Linear(X_nn.shape[1], self.hidden_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_layer_size, y_weights.shape[1]),
            torch.nn.Sigmoid(),
        ).to(self.device)
        optim = torch.optim.Adam(self.weight_estimator.parameters())
        loss_fn = torch.nn.BCELoss().to(self.device)

        # We specify the batch_size for the user when they did not specify it.
        if self.batch_size is None:
            self.batch_size = min(200, X_nn.shape[0])

        for epoch in tqdm(range(500), disable=(not self.verbose)):
            for iter in range(0, X_nn.shape[0], self.batch_size):
                optim.zero_grad()
                y_hat = self.weight_estimator(
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
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the weight of each base estimator for each sample
        """
        X_transformed = self.scaler.transform(X)

        self.weight_estimator.eval()
        with torch.no_grad():
            y_weights_hat = (
                self.weight_estimator(
                    torch.from_numpy(X_transformed).float().to(self.device)
                )
                .cpu()
                .numpy()
            )

        return y_weights_hat


class CAMSStacker(BaseEstimator):
    """
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

    """

    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        calibration_method: str = "isotonic",
        cv_calibration: int = 5,
        cv_stacking: int = 5,
        refit: bool = True,
        batch_size: Optional[int] = None,
        hidden_layer_size: int = 50,
        device: Optional[str] = None,
        verbose: bool = True,
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

        self.weight_estimator: WeightEstimator = None
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

        all_clf_truth = []
        for idx_train, idx_valid in StratifiedKFold(
            n_splits=self.cv_stacking, shuffle=False
        ).split(X, y):
            X_train, X_valid = X[idx_train], X[idx_valid]
            y_train, y_valid = y[idx_train], y[idx_valid]

            # Parallelly fit the calibrated estimators using the training and validation
            # folds.
            self.cal_estimators = Parallel()(
                delayed(
                    CalibratedClassifierCV(
                        e, method=self.calibration_method, cv=self.cv_calibration
                    ).fit
                )(X_train, y_train)
                for e in self.base_estimators
            )

            all_clf_truth_fold = self.calculate_estimator_weights(X_valid, y_valid)
            all_clf_truth.append(all_clf_truth_fold)

        y_weights = np.vstack(all_clf_truth)

        self.weight_estimator = WeightEstimator(
            hidden_layer_size=self.hidden_layer_size,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        ).fit(X, y_weights)

        # Re-fit with full-dataset if requested
        if self.refit:
            self.cal_estimators = Parallel()(
                delayed(
                    CalibratedClassifierCV(
                        e, method=self.calibration_method, cv=self.cv_calibration
                    ).fit
                )(X, y)
                for e in self.base_estimators
            )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        y_weights_hat = self.weight_estimator.predict(X)

        weighted_votes = sum(
            [
                np.multiply(e.predict_proba(X), y_weights_hat[:, [i]])
                for i, e in enumerate(self.cal_estimators)
            ]
        )

        labels = self.encoder.inverse_transform(weighted_votes).squeeze()
        return labels
