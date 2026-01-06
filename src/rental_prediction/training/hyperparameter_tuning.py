"""Hyperparameter optimization using Optuna with MLflow integration.

This module provides utilities for automated hyperparameter tuning of ML models
using Optuna's Bayesian optimization. The HyperparameterTuner class orchestrates
the optimization process and optionally logs all trials to MLflow for tracking
and comparison.

Key Features:
    - Bayesian optimization using Optuna's TPE sampler
    - Automatic MLflow logging of all trials
    - Model-agnostic interface via BaseModel protocol
    - Customizable objective metrics (RMSE, R², etc.)
    - Progress tracking and best parameter extraction

Example:
    >>> from rental_prediction.models import XGBoostModel
    >>> from rental_prediction.training.experiment_tracking import ExperimentTracker
    >>>
    >>> tracker = ExperimentTracker(experiment_name="hyperparameter-tuning")
    >>> tuner = HyperparameterTuner(
    ...     model_class=XGBoostModel,
    ...     objective_metric="rmse",
    ...     experiment_tracker=tracker
    ... )
    >>> best_params = tuner.optimize(X_train, y_train, X_val, y_val, n_trials=100)
    >>> best_model = tuner.get_best_model()
"""

import optuna
from typing import Dict, Any, Callable, Optional, Type
import numpy as np
from loguru import logger
from rental_prediction.training.experiment_tracking import ExperimentTracker
from rental_prediction.models.base_model import BaseModel


class HyperparameterTuner:
    """Hyperparameter tuning orchestrator using Optuna with MLflow integration.

    Automates the process of finding optimal hyperparameters for ML models using
    Bayesian optimization (TPE sampler). Supports any model class that implements
    the BaseModel interface and its suggest_params() classmethod.

    The tuner creates multiple trials, each training a model with different
    hyperparameters suggested by Optuna. Performance on the validation set
    determines which parameters are optimal.

    Attributes:
        model_class: The model class (type) to optimize
        objective_metric: Metric to optimize ("rmse" or "r2")
        experiment_tracker: Optional MLflow tracker for logging trials
        study: Optuna study object (created during optimization)
        best_model: Best performing model (trained after optimization)

    Example:
        >>> tuner = HyperparameterTuner(
        ...     model_class=XGBoostModel,
        ...     objective_metric="rmse",
        ...     experiment_tracker=tracker
        ... )
        >>> best_params = tuner.optimize(X_train, y_train, X_val, y_val, n_trials=50)
        >>> stats = tuner.get_study_stats()
        >>> print(f"Best RMSE: {stats['best_value']:.2f}")
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        objective_metric: str = "rmse",
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        """Initialize hyperparameter tuner.

        Args:
            model_class: Model CLASS to optimize (e.g., XGBoostModel, LightGBMModel).
                        Note: This is the CLASS itself, not an instance!
                        Usage: HyperparameterTuner(model_class=XGBoostModel)
                        NOT:   HyperparameterTuner(model_class=XGBoostModel())
            objective_metric: Metric to optimize (default: 'rmse')
            experiment_tracker: Optional MLflow experiment tracker for logging
        """
        # Store the MODEL CLASS (type), not an instance
        # We'll create many instances during optimization, each with different params
        self.model_class = model_class
        self.objective_metric = objective_metric
        self.experiment_tracker = experiment_tracker
        self.study = None
        self.best_model = None

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
        direction: str = "minimize",
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            direction: Optimization direction ('minimize' or 'maximize')
            study_name: Optional name for the Optuna study

        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            # Step 1: Get hyperparameter suggestions from the MODEL CLASS
            # This is why suggest_params() is a @classmethod - we call it on
            # the class (e.g., XGBoostModel) before any instance exists.
            # Each model type defines its own search space (XGBoost has different
            # hyperparameters than LightGBM).
            suggested_params = self.model_class.suggest_params(trial)

            # Step 2: NOW create an instance with the suggested parameters
            # We couldn't do this before getting the params from suggest_params()
            model = self.model_class(model_params=suggested_params)

            # Step 3: Train the model with the suggested hyperparameters
            model.train(X_train, y_train)

            # Step 4: Evaluate on validation set to get the objective value
            rmse, r2 = model.evaluate(X_val, y_val)

            # Log to MLflow if tracker is available
            if self.experiment_tracker:
                try:
                    with self.experiment_tracker.start_run(
                        run_name=f"trial_{trial.number}"
                    ):
                        self.experiment_tracker.log_params(
                            {
                                "trial_number": trial.number,
                                **suggested_params,
                            }
                        )
                        self.experiment_tracker.log_metrics(
                            {"rmse": rmse, "r2": r2, "trial": trial.number}
                        )
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            return rmse if self.objective_metric == "rmse" else -r2

        # Create and run study
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name or f"optimization_{self.model_class.__name__}",
        )

        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best {self.objective_metric}: {self.study.best_value}")
        logger.info(f"Best parameters: {self.study.best_params}")

        # Train final model with best parameters
        self.best_model = self.model_class(model_params=self.study.best_params)
        self.best_model.train(X_train, y_train)

        return self.study.best_params

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found during optimization.

        Returns the hyperparameter configuration that achieved the best
        objective value (lowest RMSE or highest R²) across all trials.

        Returns:
            Dictionary mapping hyperparameter names to their optimal values.
            Example: {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05}

        Raises:
            ValueError: If optimize() hasn't been called yet. Run optimization first.

        Example:
            >>> best_params = tuner.get_best_params()
            >>> print(f"Optimal learning rate: {best_params['learning_rate']}")
            >>> # Use params to train a new model
            >>> model = XGBoostModel(model_params=best_params)
        """
        if self.study is None:
            raise ValueError("No optimization study found. Run optimize() first.")
        return self.study.best_params

    def get_best_model(self):
        """Get the trained model with best hyperparameters from optimization.

        Returns the model instance that was trained with the best parameters
        found during optimization. This model is ready to use for predictions
        without additional training.

        Returns:
            Trained model instance (e.g., XGBoostModel or LightGBMModel) with
            the best hyperparameters. The model has already been fitted on the
            training data.

        Raises:
            ValueError: If optimize() hasn't been called yet. Run optimization first.

        Example:
            >>> best_model = tuner.get_best_model()
            >>> predictions = best_model.predict(X_test)
            >>> rmse, r2 = best_model.evaluate(X_test, y_test)
        """
        if self.best_model is None:
            raise ValueError("No model found. Run optimize() first.")
        return self.best_model

    def get_study_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the optimization study.

        Provides an overview of the optimization process including total
        trials run, best performance achieved, and which trial was best.

        Returns:
            Dictionary containing:
                - n_trials: Total number of trials completed
                - best_value: Best objective value achieved (RMSE or R²)
                - best_params: Best hyperparameters found
                - best_trial_number: Trial number that achieved best value

        Raises:
            ValueError: If optimize() hasn't been called yet. Run optimization first.

        Example:
            >>> stats = tuner.get_study_stats()
            >>> print(f"Ran {stats['n_trials']} trials")
            >>> print(f"Best RMSE: {stats['best_value']:.2f}")
            >>> print(f"Found in trial #{stats['best_trial_number']}")
            >>> print(f"Best params: {stats['best_params']}")
        """
        if self.study is None:
            raise ValueError("No optimization study found. Run optimize() first.")

        return {
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.study.best_trial.number,
        }
