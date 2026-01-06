"""Experiment tracking integration with MLflow.

This module provides utilities for logging experiments, metrics, and artifacts
to MLflow tracking server.
"""

import mlflow
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path
from mlflow.models import infer_signature


class ExperimentTracker:
    """MLflow experiment tracking wrapper for logging experiments and artifacts.

    Provides a simplified interface for common MLflow operations including
    parameter logging, metric tracking, model persistence, and artifact storage.
    Handles error logging and provides consistent experiment tracking across
    the training pipeline.

    Attributes:
        experiment_name: Name of the MLflow experiment

    Example:
        >>> tracker = ExperimentTracker(experiment_name="rental-prediction")
        >>> with tracker.start_run(run_name="experiment-1"):
        ...     tracker.log_params({"learning_rate": 0.1})
        ...     tracker.log_metrics({"rmse": 245.67, "r2": 0.82})
        ...     tracker.log_model(model, X_sample=X_train, y_sample=y_train)
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """Initialize experiment tracker and configure MLflow.

        Sets up MLflow tracking URI (if provided) and creates/activates the
        specified experiment. If the experiment already exists, it will be reused.

        Args:
            experiment_name: Name of the MLflow experiment. Used to group related runs.
            tracking_uri: Optional MLflow tracking server URI. If not provided, uses
                         local file-based tracking (./mlruns by default).
                         Examples: "http://localhost:5000" or "file:///path/to/mlruns"
        """
        self.experiment_name = experiment_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"Initialized experiment: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run within the current experiment.

        Creates a new run context for logging parameters, metrics, and artifacts.
        Should be used as a context manager to ensure proper cleanup.

        Args:
            run_name: Optional name for the run. If not provided, MLflow generates
                     a unique name. Use descriptive names like "xgboost-trial-1" for
                     easier identification.

        Returns:
            Active MLflow run context manager. Use with 'with' statement:
                with tracker.start_run(run_name="my-run"):
                    # Log parameters, metrics, etc.

        Example:
            >>> with tracker.start_run(run_name="baseline-experiment"):
            ...     tracker.log_params({"model": "xgboost"})
            ...     tracker.log_metrics({"rmse": 245.67})
        """
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration to the current MLflow run.

        Parameters are immutable once logged - attempting to log the same
        parameter again will raise an error. Log all configuration before
        training begins.

        Args:
            params: Dictionary mapping parameter names to values. Values can be
                   strings, numbers, or booleans. Nested dicts will be flattened.
                   Example: {"learning_rate": 0.1, "n_estimators": 100}

        Raises:
            Exception: If logging fails (e.g., no active run, parameter already logged)

        Example:
            >>> tracker.log_params({
            ...     "model_type": "xgboost",
            ...     "learning_rate": 0.1,
            ...     "max_depth": 6
            ... })
        """
        try:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log performance metrics to the current MLflow run.

        Unlike parameters, metrics can be logged multiple times with different
        values, enabling time-series tracking (e.g., loss per epoch).

        Args:
            metrics: Dictionary mapping metric names to numeric values.
                    Example: {"rmse": 245.67, "r2": 0.82, "mae": 180.5}
            step: Optional step number for time-series metrics. Use for tracking
                 metrics across epochs/iterations. If None, metrics are logged
                 as final values.

        Raises:
            Exception: If logging fails (e.g., no active run)

        Example:
            >>> # Log final metrics
            >>> tracker.log_metrics({"test_rmse": 245.67, "test_r2": 0.82})
            >>>
            >>> # Log metrics per epoch
            >>> for epoch in range(10):
            ...     tracker.log_metrics({"train_loss": loss}, step=epoch)
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        X_sample: Optional[np.ndarray] = None,
        y_sample: Optional[np.ndarray] = None,
    ) -> None:
        """Log trained model to MLflow with input/output signature and examples.

        Saves the model in MLflow's model format, which includes metadata about
        input/output schema for deployment and serving. The signature helps validate
        prediction requests at runtime.

        Args:
            model: Trained model object with sklearn-compatible API (must have
                  predict() method). Can be sklearn, xgboost, lightgbm, etc.
            artifact_path: Path within the run's artifact URI where the model
                          will be stored. Defaults to "model".
            X_sample: Optional sample input data (features) for signature inference.
                     Should be a 2D numpy array. First 5 rows will be used.
            y_sample: Optional sample output data (targets) for signature inference.
                     Should be a 1D numpy array. Used to infer output schema.

        Raises:
            Exception: If model logging fails

        Note:
            Providing X_sample and y_sample is highly recommended as it enables:
            - Input/output schema validation during deployment
            - Example inputs for testing deployed models
            - Better documentation of expected data format

        Example:
            >>> tracker.log_model(
            ...     model=trained_model.model,
            ...     X_sample=X_train[:100],
            ...     y_sample=y_train[:100]
            ... )
        """
        try:
            # Infer signature if sample data is provided
            signature = None
            input_example = None

            if X_sample is not None:
                input_example = X_sample[:5]  # Use first 5 samples as example

                if y_sample is not None:
                    # Predict on sample to get output signature
                    predictions = model.predict(X_sample[:5])
                    signature = infer_signature(X_sample[:5], predictions)
                else:
                    signature = infer_signature(X_sample[:5])

            mlflow.sklearn.log_model(
                model, artifact_path, signature=signature, input_example=input_example
            )
            logger.info(f"Logged model to {artifact_path} with signature")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log a local file as an artifact to the current MLflow run.

        Use this for logging additional files like plots, configuration files,
        reports, or any other file-based artifacts associated with the run.

        Args:
            local_path: Path to local file to upload. Must be an existing file.
            artifact_path: Optional subdirectory within the run's artifact directory
                          where the file should be stored. If None, file is stored
                          at the artifact root. Example: "plots" or "reports/summary"

        Raises:
            FileNotFoundError: If the local file doesn't exist
            Exception: If artifact upload fails

        Example:
            >>> # Log a plot
            >>> plt.savefig("model_performance.png")
            >>> tracker.log_artifact("model_performance.png", artifact_path="plots")
            >>>
            >>> # Log a config file
            >>> tracker.log_artifact("config.yaml", artifact_path="configs")
        """
        try:
            if Path(local_path).is_file():
                mlflow.log_artifact(local_path, artifact_path)
                logger.info(f"Logged artifact: {local_path}")
            else:
                raise FileNotFoundError(f"Artifact not found: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a Python dictionary as a JSON artifact.

        Convenient method for logging structured data like configuration,
        results, or metadata without creating temporary files.

        Args:
            dictionary: Python dictionary to log. Must be JSON-serializable
                       (values can be str, int, float, bool, list, dict, None).
            artifact_file: Name of the artifact file (should end with .json or .yaml).
                          Example: "config.json", "results.json", "metadata.yaml"

        Raises:
            Exception: If dictionary is not JSON-serializable or logging fails

        Example:
            >>> tracker.log_dict(
            ...     {
            ...         "preprocessing": {"scaler": "standard", "imputer": "mean"},
            ...         "features": ["area", "rooms", "construction_year"],
            ...         "n_samples": 1000
            ...     },
            ...     artifact_file="pipeline_config.json"
            ... )
        """
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.info(f"Logged dictionary to {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log dictionary: {e}")
            raise

    def end_run(self) -> None:
        """End the current MLflow run and finalize logging.

        Marks the run as finished and uploads any pending data to the tracking
        server. Should be called when done logging or used via context manager.

        Note:
            If using start_run() as a context manager (with statement), this is
            called automatically. Only call manually if not using context manager.

        Example:
            >>> # Manual run management
            >>> tracker.start_run(run_name="my-experiment")
            >>> tracker.log_params({"model": "xgboost"})
            >>> tracker.end_run()  # Must call manually
            >>>
            >>> # Context manager (recommended - auto-calls end_run)
            >>> with tracker.start_run(run_name="my-experiment"):
            ...     tracker.log_params({"model": "xgboost"})
            ...     # end_run() called automatically
        """
        mlflow.end_run()
        logger.info("Ended MLflow run")
