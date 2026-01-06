"""Training module for model training orchestration and hyperparameter tuning.

This module provides a complete training infrastructure for ML models including:

Components:
    - Trainer: End-to-end training pipeline orchestration
    - HyperparameterTuner: Automated hyperparameter optimization with Optuna
    - ExperimentTracker: MLflow integration for experiment tracking

The training module coordinates:
    - Data preprocessing and splitting
    - Model training and evaluation
    - Hyperparameter optimization
    - Experiment tracking and logging
    - Model persistence and versioning

Example:
    >>> from rental_prediction.training import Trainer, ExperimentTracker
    >>> from rental_prediction.models import XGBoostModel
    >>> from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor
    >>> from rental_prediction.config.model_config import ModelConfig
    >>>
    >>> # Setup components
    >>> model = XGBoostModel()
    >>> preprocessor = DataPreprocessor()
    >>> config = ModelConfig()
    >>> tracker = ExperimentTracker(experiment_name="rental-prediction")
    >>>
    >>> # Train model
    >>> trainer = Trainer(model, preprocessor, config, experiment_tracker=tracker)
    >>> metrics = trainer.train(data, run_name="baseline")
"""
