"""LightGBM regression model implementation for rental price prediction.

This module provides a LightGBM-based implementation of the BaseModel interface.
LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that
uses tree-based learning algorithms, optimized for speed and memory efficiency.
"""

from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import optuna

from rental_prediction.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM regression model for rental price prediction.

    Implements gradient boosting regression using the LightGBM library.
    LightGBM is particularly efficient for large datasets and provides
    faster training speed and lower memory usage compared to other GBDT tools.

    Attributes:
        model_params: Dictionary of LightGBM hyperparameters
        model: The trained LGBMRegressor instance (None until trained)
        best_params: Best hyperparameters from optimization (None until optimized)

    Example:
        >>> model = LightGBMModel(model_params={'n_estimators': 100, 'num_leaves': 31})
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> rmse, r2 = model.evaluate(X_test, y_test)
    """

    def __init__(self, model_params: Dict[str, Any] = None):
        """Initialize LightGBM model with specified hyperparameters.

        Args:
            model_params: Dictionary of LightGBM hyperparameters. Common parameters:
                - n_estimators: Number of boosting rounds
                - max_depth: Maximum tree depth (use -1 for no limit)
                - learning_rate: Boosting learning rate
                - num_leaves: Maximum number of leaves in one tree
                - subsample: Subsample ratio of training instances
                - colsample_bytree: Subsample ratio of columns
                - min_child_samples: Minimum number of data in one leaf
                - reg_alpha: L1 regularization term
                - reg_lambda: L2 regularization term
                Defaults to empty dict (uses LightGBM defaults).
        """
        super().__init__(model_params)
        self.model = None

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters using Optuna trial.

        Defines the hyperparameter search space specific to LightGBM models.
        This is called on the CLASS (LightGBMModel.suggest_params(trial)),
        not on an instance, because we need these parameters to CREATE
        the instance in the first place.

        Args:
            trial: Optuna trial object for suggesting parameters

        Returns:
            Dictionary of suggested LightGBM hyperparameters
        """
        return {
            # Tree structure parameters
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),  # LightGBM-specific

            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),

            # Sampling parameters to prevent overfitting
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),

            # Regularization parameters
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),  # L1 regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),  # L2 regularization
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the LightGBM model on the provided training data.

        Creates an LGBMRegressor instance with the configured hyperparameters
        and fits it to the training data.

        Args:
            X_train: Training features as a 2D numpy array of shape (n_samples, n_features)
            y_train: Training labels as a 1D numpy array of shape (n_samples,)

        Note:
            After training, the model is stored in self.model and can be used
            for making predictions via the predict() method.
        """
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained LightGBM model.

        Args:
            X: Features to predict on as a 2D numpy array of shape (n_samples, n_features)

        Returns:
            Predicted rental prices as a 1D numpy array of shape (n_samples,)

        Raises:
            ValueError: If the model has not been trained yet. Call train() first.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna.

        Performs Bayesian optimization to find the best hyperparameters that
        minimize RMSE on the validation set. After optimization completes,
        the model is retrained with the best parameters.

        Args:
            X_train: Training features as a 2D numpy array of shape (n_samples, n_features)
            y_train: Training labels as a 1D numpy array of shape (n_samples,)
            X_val: Validation features as a 2D numpy array of shape (n_samples, n_features)
            y_val: Validation labels as a 1D numpy array of shape (n_samples,)
            n_trials: Number of optimization trials to run. More trials may find
                     better parameters but takes longer. Defaults to 100.

        Returns:
            Dictionary containing the best hyperparameters found during optimization.
            These parameters are also stored in self.best_params and applied to
            self.model_params.

        Note:
            This method trains the model n_trials + 1 times (once per trial, plus
            a final training with best params). For large datasets, consider using
            the HyperparameterTuner class instead, which integrates with MLflow.
        """

        def objective(trial):
            """Optuna objective function to minimize RMSE."""
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            }

            self.model_params = params
            self.train(X_train, y_train)
            rmse, _ = self.evaluate(X_val, y_val)
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.model_params = self.best_params
        self.train(X_train, y_train)
        return self.best_params
