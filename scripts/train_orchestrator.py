#!/usr/bin/env python3
"""Training orchestrator for continuous training pipeline.

This script coordinates model training, comparison, and generates reports
for automated model promotion decisions.

Usage:
    python scripts/train_orchestrator.py --model-type both --n-trials 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger

from rental_prediction.data import CSVLoader
from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor
from rental_prediction.models import XGBoostModel, LightGBMModel
from rental_prediction.training.trainer import Trainer
from rental_prediction.training.experiment_tracking import ExperimentTracker
from rental_prediction.config.model_config import ModelConfig
from rental_prediction.utils.model_registry import ModelRegistry


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train rental prediction models for production"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="both",
        choices=["xgboost", "lightgbm", "both"],
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of hyperparameter tuning trials"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/rent_apartments.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/training_report.json",
        help="Output path for training report",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI",
    )
    return parser.parse_args()


def train_model(
    model_class,
    model_name: str,
    data,
    config: ModelConfig,
    tracker: ExperimentTracker,
    registry: ModelRegistry,
) -> Dict[str, Any]:
    """Train a single model and return metrics.

    Args:
        model_class: Model class to instantiate
        model_name: Name for the model
        data: Training data
        config: Model configuration
        tracker: MLflow experiment tracker
        registry: Model registry for saving

    Returns:
        Dictionary with model metrics and metadata
    """
    logger.info(f"Training {model_name}...")

    # Initialize components
    preprocessor = DataPreprocessor()
    model = model_class(
        model_params={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
    )

    # Train
    trainer = Trainer(
        model=model,
        preprocessor=preprocessor,
        config=config,
        experiment_tracker=tracker,
        model_registry=registry,
    )

    metrics = trainer.train(data, run_name=f"{model_name}-production")

    result = {
        "name": model_name,
        "model_class": model_class.__name__,
        "val_rmse": metrics["val_rmse"],
        "val_r2": metrics["val_r2"],
        "test_rmse": metrics["test_rmse"],
        "test_r2": metrics["test_r2"],
        "trained_at": datetime.now().isoformat(),
    }

    logger.info(
        f"✅ {model_name} - Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}"
    )

    return result


def main():
    """Main training orchestration workflow."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("TRAINING ORCHESTRATOR - Production Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Hyperparameter trials: {args.n_trials}")
    logger.info(f"Data path: {args.data_path}")

    # Load data
    logger.info("Loading and validating data...")
    loader = CSVLoader(args.data_path, validate=True)
    data = loader.load()
    logger.info(f"Loaded {len(data)} records")

    # Initialize shared components
    config = ModelConfig()
    tracker = ExperimentTracker(
        experiment_name="rental-prediction-production",
        tracking_uri=args.mlflow_uri,
    )
    registry = ModelRegistry(registry_path=Path("models"))

    # Determine which models to train
    models_to_train = []
    if args.model_type in ["xgboost", "both"]:
        models_to_train.append((XGBoostModel, "xgboost"))
    if args.model_type in ["lightgbm", "both"]:
        models_to_train.append((LightGBMModel, "lightgbm"))

    # Train models
    results: List[Dict[str, Any]] = []
    for model_class, model_name in models_to_train:
        try:
            result = train_model(
                model_class=model_class,
                model_name=model_name,
                data=data,
                config=config,
                tracker=tracker,
                registry=registry,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue

    if not results:
        raise RuntimeError("No models were successfully trained")

    # Select best model (lowest test RMSE)
    best_model = min(results, key=lambda x: x["test_rmse"])
    logger.info(f"\n🏆 Best model: {best_model['name']} (RMSE: {best_model['test_rmse']:.4f})")

    # Generate report
    report = {
        "training_timestamp": datetime.now().isoformat(),
        "data_path": args.data_path,
        "data_records": len(data),
        "models_trained": [r["name"] for r in results],
        "best_model": best_model,
        "all_results": results,
        "config": {
            "n_trials": args.n_trials,
            "test_size": config.test_size,
            "val_size": config.val_size,
        },
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📊 Training report saved to: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    for result in results:
        logger.info(
            f"  {result['name']:12} - RMSE: {result['test_rmse']:7.4f}, R²: {result['test_r2']:.4f}"
        )

    logger.info("=" * 80)
    logger.info("✅ Training orchestration completed successfully")


if __name__ == "__main__":
    main()
