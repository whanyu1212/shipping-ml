"""Base model interface for rental price prediction models.

This module defines the abstract base class that all prediction models must implement.
It provides a consistent interface for training, prediction, hyperparameter optimization,
and evaluation across different model types (XGBoost, LightGBM, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score


class BaseModel(ABC):
    """Abstract base class for rental price prediction models.

    Defines the interface that all concrete model implementations must follow.
    Provides shared functionality for model evaluation and hyperparameter suggestion.

    Attributes:
        model_params: Dictionary of model hyperparameters
        model: The underlying trained model instance (None until trained)
        best_params: Best hyperparameters found during optimization (None until optimized)
    """

    def __init__(self, model_params: Dict[str, Any] = None):
        """Initialize the base model.

        Args:
            model_params: Dictionary of model hyperparameters. Defaults to empty dict.
                         Specific parameters depend on the concrete model type.
        """
        self.model_params = model_params or {}
        self.model = None
        self.best_params = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the provided training data.

        This method must be implemented by all concrete model subclasses.
        It should fit the model to the training data using the configured
        hyperparameters in self.model_params.

        Args:
            X_train: Training features as a 2D numpy array of shape (n_samples, n_features)
            y_train: Training labels as a 1D numpy array of shape (n_samples,)

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                                by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided features.

        This method must be implemented by all concrete model subclasses.
        It should return predictions using the trained model.

        Args:
            X: Features to predict on as a 2D numpy array of shape (n_samples, n_features)

        Returns:
            Predicted values as a 1D numpy array of shape (n_samples,)

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                                by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for this model using Optuna trial.

        This is a CLASSMETHOD because:
        1. It's called BEFORE instantiation - we need the hyperparameter search
           space to create model instances during optimization
        2. The search space is metadata about the MODEL TYPE (e.g., XGBoost),
           not about a specific model instance
        3. Enables generic optimization: HyperparameterTuner can work with any
           model class by calling ModelClass.suggest_params(trial) without
           needing to instantiate it first

        Usage pattern in hyperparameter tuning:
            # Step 1: Get suggested params from the CLASS (no instance yet)
            params = XGBoostModel.suggest_params(trial)

            # Step 2: Create instance with those params
            model = XGBoostModel(model_params=params)

            # Step 3: Train and evaluate
            model.train(X_train, y_train)

        Args:
            trial: Optuna trial object for suggesting parameters

        Returns:
            Dictionary of suggested hyperparameters for this model type

        Note:
            Subclasses should override this to define their hyperparameter
            search space. The base implementation returns empty dict.
        """
        # Base implementation - subclasses define their specific search spaces
        return {}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model's performance on provided data.

        Calculates RMSE (Root Mean Squared Error) and R² (coefficient of determination)
        metrics to assess model performance. RMSE measures prediction error in the
        same units as the target variable, while R² indicates the proportion of
        variance explained by the model (0-1, higher is better).

        Args:
            X: Features to evaluate on as a 2D numpy array of shape (n_samples, n_features)
            y: True labels as a 1D numpy array of shape (n_samples,)

        Returns:
            Tuple containing:
                - rmse: Root mean squared error (float)
                - r2: R² score, coefficient of determination (float)

        Raises:
            ValueError: If the model has not been trained yet (via predict method)
        """
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return rmse, r2
