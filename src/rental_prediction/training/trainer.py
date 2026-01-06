"""Training orchestration for ML models.

This module handles the end-to-end training workflow including:
- Data loading and preprocessing
- Train/val/test split
- Model training
- Evaluation
- Model persistence
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

from rental_prediction.training.experiment_tracking import ExperimentTracker
from rental_prediction.utils.model_registry import ModelRegistry


class Trainer:
    """Orchestrates the end-to-end training workflow for ML models.

    Coordinates the complete training pipeline including data preprocessing,
    train/validation/test splitting, model training, evaluation, experiment
    tracking with MLflow, and model persistence to a registry.

    Attributes:
        model: Model instance to train (must implement BaseModel interface)
        preprocessor: Data preprocessor for transforming raw data
        config: Configuration with hyperparameters and split ratios
        experiment_tracker: Optional MLflow tracker for experiment logging
        model_registry: Optional registry for model versioning and storage
        metrics: Dictionary storing evaluation metrics after training

    Example:
        >>> from rental_prediction.models import XGBoostModel
        >>> from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor
        >>> from rental_prediction.config.model_config import ModelConfig
        >>>
        >>> model = XGBoostModel(model_params={"n_estimators": 100})
        >>> preprocessor = DataPreprocessor()
        >>> config = ModelConfig(test_size=0.2, val_size=0.2)
        >>> tracker = ExperimentTracker(experiment_name="my-experiment")
        >>>
        >>> trainer = Trainer(model, preprocessor, config, experiment_tracker=tracker)
        >>> metrics = trainer.train(raw_data, run_name="baseline")
        >>> print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    """

    def __init__(
        self,
        model,
        preprocessor,
        config,
        experiment_tracker: Optional[ExperimentTracker] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """Initialize the trainer with model, preprocessor, and configuration.

        Args:
            model: Model instance to train. Must implement train(), predict(),
                  and evaluate() methods (BaseModel interface).
            preprocessor: Data preprocessor instance for transforming raw data.
                         Must implement process() method.
            config: Configuration object with training parameters. Should have
                   test_size and val_size attributes for data splitting.
            experiment_tracker: Optional ExperimentTracker for logging to MLflow.
                               If provided, all training metrics, parameters, and
                               models will be logged automatically.
            model_registry: Optional ModelRegistry for saving trained models.
                           If provided, models will be versioned and stored
                           after successful training.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.model_registry = model_registry
        self.metrics = {}

    def _split_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split preprocessed data into train, validation, and test sets.

        Performs a two-stage split:
        1. First separates test set (holdout for final evaluation)
        2. Then splits remaining data into train and validation sets

        The validation set is used for hyperparameter tuning and early stopping,
        while the test set provides unbiased final performance metrics.

        Args:
            data: Preprocessed DataFrame containing features and target column 'rent'.
                 All preprocessing (encoding, scaling, etc.) should be complete.

        Returns:
            Tuple of 6 numpy arrays:
                - X_train: Training features (n_train_samples, n_features)
                - X_val: Validation features (n_val_samples, n_features)
                - X_test: Test features (n_test_samples, n_features)
                - y_train: Training labels (n_train_samples,)
                - y_val: Validation labels (n_val_samples,)
                - y_test: Test labels (n_test_samples,)

        Note:
            Uses random_state=42 for reproducible splits. Split sizes are
            determined by config.test_size and config.val_size.
        """
        logger.info("Splitting data into train/val/test sets")

        # Separate features and target
        X = data.drop(columns=["rent"])
        y = data["rent"]

        # First split: separate test set (holdout for final evaluation)
        test_size = getattr(self.config, "test_size", 0.2)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Second split: separate train and validation from remaining data
        val_size = getattr(self.config, "val_size", 0.2)
        # Calculate validation size relative to temp set
        # E.g., if test=0.2 and val=0.2, need val/(1-test) = 0.2/0.8 = 0.25
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )

        logger.info(
            f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(
        self, data: pd.DataFrame, run_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Execute the complete end-to-end training pipeline.

        Orchestrates all training steps:
        1. Data preprocessing (via preprocessor)
        2. Train/validation/test splitting
        3. Model training on training set
        4. Evaluation on validation and test sets
        5. Experiment tracking (if tracker provided)
        6. Model persistence (if registry provided)

        Args:
            data: Raw input data as a pandas DataFrame. Should contain all
                 required features plus the 'rent' target column. Will be
                 preprocessed automatically.
            run_name: Optional name for the MLflow run. Used for organizing
                     experiments. Example: "xgboost-baseline", "lightgbm-trial-1"

        Returns:
            Dictionary containing evaluation metrics:
                - val_rmse: Validation set RMSE
                - val_r2: Validation set R² score
                - test_rmse: Test set RMSE (final performance)
                - test_r2: Test set R² score (final performance)

        Raises:
            Exception: If any step of the pipeline fails (preprocessing, training,
                      evaluation, or persistence)

        Note:
            This method automatically handles MLflow run lifecycle. If an
            experiment_tracker is configured, it will:
            - Start a new run
            - Log parameters, metrics, and the model
            - End the run (even if errors occur)

        Example:
            >>> trainer = Trainer(model, preprocessor, config)
            >>> metrics = trainer.train(raw_data, run_name="baseline-experiment")
            >>> print(f"Final Test RMSE: {metrics['test_rmse']:.2f}")
            >>> print(f"Final Test R²: {metrics['test_r2']:.3f}")
        """
        logger.info("Starting training pipeline")

        # Start MLflow run if tracker is available
        if self.experiment_tracker:
            self.experiment_tracker.start_run(run_name=run_name)

        try:
            # Step 1: Preprocess data
            logger.info("Preprocessing data")
            processed_data = self.preprocessor.process(data)

            # Step 2: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
                processed_data
            )

            # Step 3: Train model
            logger.info("Training model")
            self.model.train(X_train.values, y_train.values)

            # Log model parameters
            if self.experiment_tracker:
                self.experiment_tracker.log_params(
                    {
                        "model_type": self.model.__class__.__name__,
                        **self.model.model_params,
                    }
                )

            # Step 4: Evaluate on validation set
            logger.info("Evaluating on validation set")
            val_rmse, val_r2 = self.model.evaluate(X_val.values, y_val.values)
            self.metrics["val_rmse"] = val_rmse
            self.metrics["val_r2"] = val_r2

            logger.info(f"Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

            # Step 5: Evaluate on test set
            logger.info("Evaluating on test set")
            test_rmse, test_r2 = self.model.evaluate(X_test.values, y_test.values)
            self.metrics["test_rmse"] = test_rmse
            self.metrics["test_r2"] = test_r2

            logger.info(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

            # Log metrics to MLflow
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(self.metrics)
                # Log model with signature and input example
                self.experiment_tracker.log_model(
                    self.model.model,
                    X_sample=X_train.values,
                    y_sample=y_train.values,
                )

            # Step 6: Save model to registry
            if self.model_registry:
                logger.info("Saving model to registry")
                model_name = f"{self.model.__class__.__name__}"
                version = self.model_registry.save_model(
                    model=self.model.model,
                    model_name=model_name,
                    metadata={
                        "model_params": self.model.model_params,
                        "metrics": self.metrics,
                        "config": {
                            "test_size": getattr(self.config, "test_size", 0.2),
                            "val_size": getattr(self.config, "val_size", 0.2),
                        },
                    },
                )
                logger.info(f"Model saved as {model_name} version {version}")

            logger.info("Training pipeline completed successfully")

            return self.metrics

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

        finally:
            # End MLflow run
            if self.experiment_tracker:
                self.experiment_tracker.end_run()

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the trained model on new/unseen data.

        Useful for evaluating model performance on hold-out datasets or
        new data after deployment. Applies the same preprocessing pipeline
        used during training before evaluation.

        Args:
            data: Raw data to evaluate on as a pandas DataFrame. Should have
                 the same schema as training data, including the 'rent' target
                 column. Will be preprocessed using the same transformations
                 as training data.

        Returns:
            Dictionary containing evaluation metrics:
                - rmse: Root Mean Squared Error on the provided data
                - r2: R² score (coefficient of determination)

        Raises:
            ValueError: If the model hasn't been trained yet
            Exception: If preprocessing or evaluation fails

        Note:
            This method uses the preprocessor configured during initialization,
            so the same feature engineering and transformations are applied.

        Example:
            >>> # Evaluate on hold-out validation data
            >>> val_metrics = trainer.evaluate(validation_data)
            >>> print(f"Validation RMSE: {val_metrics['rmse']:.2f}")
            >>>
            >>> # Evaluate on new production data
            >>> prod_metrics = trainer.evaluate(production_data)
            >>> if prod_metrics['rmse'] > threshold:
            ...     print("Warning: Model performance degraded!")
        """
        logger.info("Evaluating model on provided data")

        # Preprocess data using the same pipeline as training
        processed_data = self.preprocessor.process(data)

        # Separate features and target
        X = processed_data.drop(columns=["rent"]).values
        y = processed_data["rent"].values

        # Evaluate model performance
        rmse, r2 = self.model.evaluate(X, y)

        metrics = {"rmse": rmse, "r2": r2}
        logger.info(f"Evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}")

        return metrics
