"""Pydantic schemas for API request and response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ApartmentFeatures(BaseModel):
    """Input features for a single apartment prediction."""

    area: float = Field(..., gt=0, description="Area in square meters")
    rooms: int = Field(..., ge=1, le=10, description="Number of rooms")
    construction_year: int = Field(..., ge=0, description="Year of construction")
    balcony: str = Field(..., description="Balcony type (yes/no)")
    parking: str = Field(..., description="Parking availability (yes/no)")
    furnished: str = Field(..., description="Furnished status (yes/no)")
    garage: str = Field(..., description="Garage availability (yes/no)")
    storage: str = Field(..., description="Storage availability (yes/no)")
    garden: str = Field(..., description="Garden area (e.g., '100 m²', 'Not present')")

    class Config:
        schema_extra = {
            "example": {
                "area": 85.5,
                "rooms": 3,
                "construction_year": 2010,
                "balcony": "yes",
                "parking": "yes",
                "furnished": "no",
                "garage": "no",
                "storage": "yes",
                "garden": "50 m²",
            }
        }


class PredictionRequest(BaseModel):
    """Request body for single prediction."""

    features: ApartmentFeatures


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""

    apartments: List[ApartmentFeatures] = Field(
        ..., min_items=1, max_items=1000, description="List of apartments to predict"
    )


class PredictionResponse(BaseModel):
    """Response for single prediction."""

    predicted_rent: float = Field(..., description="Predicted monthly rent")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")

    class Config:
        schema_extra = {
            "example": {
                "predicted_rent": 1250.50,
                "model_name": "XGBoostModel",
                "model_version": "20240115_140530",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[float] = Field(..., description="List of predicted rents")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    count: int = Field(..., description="Number of predictions made")


class ModelInfo(BaseModel):
    """Information about the current production model."""

    model_name: str
    model_class: str
    version: Optional[str] = None
    test_rmse: float
    test_r2: float
    trained_at: str
    available_features: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_info: Optional[ModelInfo] = None
