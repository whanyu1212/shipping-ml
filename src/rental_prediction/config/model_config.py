from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model training and hyperparameter tuning."""

    model_name: Literal["xgboost", "lightgbm"] = Field(
        default="xgboost",
        description="Model type to use for training (xgboost or lightgbm)",
    )
    n_trials: int = Field(
        default=20,
        gt=0,
        le=1000,
        description="Number of hyperparameter tuning trials",
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Test set size (0-1)"
    )
    val_size: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Validation set size (0-1)"
    )
