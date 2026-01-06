#!/usr/bin/env python3
"""Example: Complete training pipeline with MLflow tracking.

This example demonstrates the full training workflow including:
- Data loading and preprocessing
- Model training with XGBoost and LightGBM
- MLflow experiment tracking
- Model evaluation

Usage:
    python examples/train_model.py
"""

from rental_prediction.data import CSVLoader
from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor
from rental_prediction.models import XGBoostModel, LightGBMModel
from rental_prediction.training.trainer import Trainer
from rental_prediction.training.experiment_tracking import ExperimentTracker
from rental_prediction.config.model_config import ModelConfig


def train_xgboost():
    """Train XGBoost model with MLflow tracking."""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    loader = CSVLoader("data/rent_apartments.csv", validate=True)
    data = loader.load()
    print(f"   Loaded {len(data)} records")

    # Initialize components
    print("\n2. Initializing components...")
    preprocessor = DataPreprocessor()
    model = XGBoostModel(
        model_params={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
    )
    config = ModelConfig()
    tracker = ExperimentTracker(
        experiment_name="rental-prediction-example", tracking_uri="./mlruns"
    )

    # Train model
    print("\n3. Training model...")
    trainer = Trainer(
        model=model,
        preprocessor=preprocessor,
        config=config,
        experiment_tracker=tracker,
    )

    metrics = trainer.train(data, run_name="xgboost-baseline")

    # Display results
    print("\n" + "=" * 80)
    print("XGBOOST RESULTS")
    print("=" * 80)
    print(f"✅ Validation RMSE: {metrics['val_rmse']:.4f}")
    print(f"✅ Validation R²:   {metrics['val_r2']:.4f}")
    print(f"✅ Test RMSE:       {metrics['test_rmse']:.4f}")
    print(f"✅ Test R²:         {metrics['test_r2']:.4f}")
    print(f"\n📊 View results in MLflow UI: mlflow ui --backend-store-uri ./mlruns")

    return metrics


def train_lightgbm():
    """Train LightGBM model with MLflow tracking."""
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM MODEL")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    loader = CSVLoader("data/rent_apartments.csv", validate=True)
    data = loader.load()
    print(f"   Loaded {len(data)} records")

    # Initialize components
    print("\n2. Initializing components...")
    preprocessor = DataPreprocessor()
    model = LightGBMModel(
        model_params={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
    )
    config = ModelConfig()
    tracker = ExperimentTracker(
        experiment_name="rental-prediction-example", tracking_uri="./mlruns"
    )

    # Train model
    print("\n3. Training model...")
    trainer = Trainer(
        model=model,
        preprocessor=preprocessor,
        config=config,
        experiment_tracker=tracker,
    )

    metrics = trainer.train(data, run_name="lightgbm-baseline")

    # Display results
    print("\n" + "=" * 80)
    print("LIGHTGBM RESULTS")
    print("=" * 80)
    print(f"✅ Validation RMSE: {metrics['val_rmse']:.4f}")
    print(f"✅ Validation R²:   {metrics['val_r2']:.4f}")
    print(f"✅ Test RMSE:       {metrics['test_rmse']:.4f}")
    print(f"✅ Test R²:         {metrics['test_r2']:.4f}")
    print(f"\n📊 View results in MLflow UI: mlflow ui --backend-store-uri ./mlruns")

    return metrics


def compare_models():
    """Train and compare both models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    xgb_metrics = train_xgboost()
    lgb_metrics = train_lightgbm()

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nTest Set Performance:")
    print(
        f"  XGBoost  - RMSE: {xgb_metrics['test_rmse']:.4f}, R²: {xgb_metrics['test_r2']:.4f}"
    )
    print(
        f"  LightGBM - RMSE: {lgb_metrics['test_rmse']:.4f}, R²: {lgb_metrics['test_r2']:.4f}"
    )

    winner = (
        "XGBoost" if xgb_metrics["test_rmse"] < lgb_metrics["test_rmse"] else "LightGBM"
    )
    print(f"\n🏆 Best Model: {winner}")


def main():
    """Run the training example."""
    import sys

    print("=" * 80)
    print("RENTAL PREDICTION - TRAINING EXAMPLE")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Complete training pipeline")
    print("  2. MLflow experiment tracking")
    print("  3. Model comparison (XGBoost vs LightGBM)")
    print("\nChoose an option:")
    print("  1 - Train XGBoost only")
    print("  2 - Train LightGBM only")
    print("  3 - Train and compare both models")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        train_xgboost()
    elif choice == "2":
        train_lightgbm()
    elif choice == "3":
        compare_models()
    else:
        print("Invalid choice. Running full comparison...")
        compare_models()


if __name__ == "__main__":
    main()
