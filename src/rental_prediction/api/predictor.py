"""Prediction service for loading models and making predictions."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from loguru import logger

from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor
from rental_prediction.utils.model_registry import ModelRegistry


class PredictionService:
    """Service for loading models and making predictions."""

    def __init__(
        self,
        model_registry_path: str = "models",
        use_production_baseline: bool = True,
    ):
        """Initialize prediction service.

        Args:
            model_registry_path: Path to model registry
            use_production_baseline: If True, load from production_baseline.json
        """
        self.registry = ModelRegistry(registry_path=Path(model_registry_path))
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.model_metadata = None
        self.use_production_baseline = use_production_baseline

    def load_model(self) -> None:
        """Load the model from the registry.

        Raises:
            FileNotFoundError: if no model is found in the registry
            FileNotFoundError: if production baseline metadata is not found
        """
        try:
            if self.use_production_baseline:
                # Load production baseline metadata
                baseline_path = Path("registry/production_baseline.json")
                if not baseline_path.exists():
                    raise FileNotFoundError(
                        "Production baseline not found. Train a model first."
                    )

                with open(baseline_path, "r") as f:
                    self.model_metadata = json.load(f)

                model_name = self.model_metadata["name"]
                logger.info(f"Loading production model: {model_name}")

                # Load the latest version of this model from registry
                self.model = self.registry.load_model(model_name)
            else:
                # Load latest model from registry
                models = self.registry.list_models()
                if not models:
                    raise FileNotFoundError("No models found in registry")

                model_name = models[0]
                self.model = self.registry.load_model(model_name)
                self.model_metadata = self.registry.load_metadata(model_name)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_single(self, features: Dict[str, Any]) -> float:
        """Make prediction for a single apartment.

        Args:
            features (Dict[str, Any]): Dictionary of apartment features

        Returns:
            float: Predicted rent value
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Preprocess
        processed = self.preprocessor.process(df)

        # Make prediction
        prediction = self.model.predict(processed.values)

        return float(prediction[0])

    def predict_batch(self, apartments: List[Dict[str, Any]]) -> List[float]:
        """Make predictions for multiple apartments.

        Args:
            apartments (List[Dict[str, Any]]): List of apartment feature dictionaries

        Returns:
            List[float]: List of predicted rent values
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to DataFrame
        df = pd.DataFrame(apartments)

        # Preprocess
        processed = self.preprocessor.process(df)

        # Make predictions
        predictions = self.model.predict(processed.values)

        return predictions.tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Dict[str, Any]: Dictionary with model metadata
        """
        if self.model is None or self.model_metadata is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return {
            "model_name": self.model_metadata.get("name", "unknown"),
            "model_class": self.model_metadata.get("model_class", "unknown"),
            "version": self.model_metadata.get("trained_at", "unknown"),
            "test_rmse": self.model_metadata.get("test_rmse", 0.0),
            "test_r2": self.model_metadata.get("test_r2", 0.0),
            "trained_at": self.model_metadata.get("trained_at", "unknown"),
            "available_features": [
                "area",
                "rooms",
                "construction_year",
                "balcony",
                "parking",
                "furnished",
                "garage",
                "storage",
                "garden",
            ],
        }

    def is_ready(self) -> bool:
        """Check if the service is ready to make predictions.

        Returns:
            bool: True if model is loaded and ready
        """
        return self.model is not None
