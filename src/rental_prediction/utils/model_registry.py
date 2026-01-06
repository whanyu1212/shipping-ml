"""Model registry for saving, loading, and versioning trained models.

This module provides utilities for model persistence and management.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from loguru import logger


class ModelRegistry:
    """Handle model persistence and versioning."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("models")
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, model_name: str, version: str) -> Path:
        """Get the directory path for a specific model version.

        Args:
            model_name: Name of the model
            version: Version string

        Returns:
            Path to the model version directory
        """
        return self.registry_path / model_name / version

    def _generate_version(self, model_name: str) -> str:
        """Generate a new version string based on timestamp.

        Args:
            model_name: Name of the model

        Returns:
            Version string in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a trained model to the registry.

        Args:
            model: Trained model object to save
            model_name: Name of the model
            version: Optional version string. If None, auto-generates timestamp version
            metadata: Optional metadata dict to save with the model

        Returns:
            Version string of the saved model
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)

        # Create model directory
        model_dir = self._get_model_dir(model_name, version)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model using pickle
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        metadata = metadata or {}
        metadata.update(
            {
                "model_name": model_name,
                "version": version,
                "saved_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
            }
        )

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model {model_name} version {version} to {model_dir}")
        return version

    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load a model from the registry.

        Args:
            model_name: Name of the model
            version: Optional version string. If None, loads the latest version

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model or version doesn't exist
        """
        # If version not specified, get latest
        if version is None:
            versions = self.list_versions(model_name)
            if not versions:
                raise FileNotFoundError(f"No versions found for model: {model_name}")
            version = versions[-1]  # Latest version

        # Load model
        model_dir = self._get_model_dir(model_name, version)
        model_path = model_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_name} version {version}"
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Loaded model {model_name} version {version}")
        return model

    def load_metadata(
        self, model_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load metadata for a model version.

        Args:
            model_name: Name of the model
            version: Optional version string. If None, loads latest version metadata

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If metadata doesn't exist
        """
        # If version not specified, get latest
        if version is None:
            versions = self.list_versions(model_name)
            if not versions:
                raise FileNotFoundError(f"No versions found for model: {model_name}")
            version = versions[-1]

        metadata_path = self._get_model_dir(model_name, version) / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for {model_name} v{version}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_models(self) -> List[str]:
        """List all registered model names.

        Returns:
            List of model names
        """
        if not self.registry_path.exists():
            return []

        models = [
            d.name for d in self.registry_path.iterdir() if d.is_dir()
        ]
        return sorted(models)

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Sorted list of version strings
        """
        model_path = self.registry_path / model_name

        if not model_path.exists():
            return []

        versions = [
            v.name for v in model_path.iterdir() if v.is_dir()
        ]
        return sorted(versions)

    def register_to_mlflow(
        self, model: Any, model_name: str, run_id: Optional[str] = None, **kwargs
    ) -> None:
        """Register model to MLflow model registry.

        Args:
            model: Trained model object
            model_name: Name for the model in MLflow registry
            run_id: Optional MLflow run ID. If None, uses current active run
            **kwargs: Additional arguments to pass to mlflow.sklearn.log_model

        Raises:
            ValueError: If no active MLflow run and run_id not provided
        """
        if run_id is None and not mlflow.active_run():
            raise ValueError(
                "No active MLflow run. Either start a run or provide run_id"
            )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            **kwargs,
        )

        logger.info(f"Registered model {model_name} to MLflow registry")
