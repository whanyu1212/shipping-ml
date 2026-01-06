"""Rental Prediction Package.

A comprehensive ML package for rental price prediction demonstrating
production-ready ML patterns and workflows.
"""

# Configuration
from .config import ModelConfig, Settings  # noqa: F401

# Data
from .data import DfSchema  # noqa: F401

# Models
from .models import BaseModel, XGBoostModel, LightGBMModel  # noqa: F401

# Preprocessing
from .preprocessor import (  # noqa: F401
    ColEncoder,
    ColumnEncoderConfig,
    DataHandler,
    DataPreprocessor,
    GardenAreaParser,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "ModelConfig",
    "Settings",
    # Data
    "DfSchema",
    # Models
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    # Preprocessing
    "ColEncoder",
    "ColumnEncoderConfig",
    "DataHandler",
    "DataPreprocessor",
    "GardenAreaParser",
]
