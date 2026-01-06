"""FastAPI application for rental prediction model serving.

Run with:
    uvicorn rental_prediction.api.main:app --reload

Or in production:
    uvicorn rental_prediction.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from rental_prediction.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
)
from rental_prediction.api.predictor import PredictionService

# Global prediction service instance
prediction_service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    # Startup: Load model
    logger.info("Starting up API server...")
    try:
        prediction_service.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("API will start but predictions will fail until model is loaded")

    yield

    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Rental Prediction API",
    description="Production ML API for predicting apartment rental prices",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware - Controls which websites can call this API from browsers
# CORS (Cross-Origin Resource Sharing) is a browser security feature that blocks
# JavaScript on one domain from calling APIs on a different domain unless explicitly allowed
app.add_middleware(
    CORSMiddleware,
    # allow_origins: Which domains can call this API from a browser
    # ["*"] = Allow ALL websites (convenient for development, INSECURE for production)
    # Production should use: ["https://yourdomain.com", "https://app.yourdomain.com"]
    allow_origins=["*"],  # TODO: Restrict to specific origins before production deployment

    # allow_credentials: Whether browsers can send cookies/auth headers with requests
    # True = Allow authentication credentials (needed if using cookies or Authorization header)
    allow_credentials=True,

    # allow_methods: Which HTTP methods are allowed from browsers
    # ["*"] = Allow all (GET, POST, PUT, DELETE, PATCH, OPTIONS, etc.)
    # Production could restrict to: ["GET", "POST"] if that's all you need
    allow_methods=["*"],

    # allow_headers: Which request headers browsers can send
    # ["*"] = Allow all custom headers
    # Production could restrict to: ["Content-Type", "Authorization"]
    allow_headers=["*"],
)


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Rental Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    is_ready = prediction_service.is_ready()

    response = {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_info": None,
    }

    if is_ready:
        try:
            response["model_info"] = prediction_service.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")

    return response


@app.get("/model/info", response_model=ModelInfo, status_code=status.HTTP_200_OK)
async def get_model_info():
    """Get information about the currently loaded model."""
    if not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service not ready.",
        )

    try:
        return prediction_service.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: PredictionRequest):
    """Make a single rental price prediction.

    Args:
        request: Prediction request with apartment features

    Returns:
        Predicted rental price with model information
    """
    if not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service not ready.",
        )

    try:
        # Convert Pydantic model to dict
        features = request.features.dict()

        # Make prediction
        predicted_rent = prediction_service.predict_single(features)

        # Get model info
        model_info = prediction_service.get_model_info()

        return PredictionResponse(
            predicted_rent=predicted_rent,
            model_name=model_info["model_name"],
            model_version=model_info["version"],
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch rental price predictions.

    Args:
        request: Batch prediction request with list of apartments

    Returns:
        List of predicted rental prices with model information
    """
    if not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service not ready.",
        )

    try:
        # Convert Pydantic models to dicts
        apartments = [apt.dict() for apt in request.apartments]

        # Make predictions
        predictions = prediction_service.predict_batch(apartments)

        # Get model info
        model_info = prediction_service.get_model_info()

        return BatchPredictionResponse(
            predictions=predictions,
            model_name=model_info["model_name"],
            model_version=model_info["version"],
            count=len(predictions),
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post("/model/reload", status_code=status.HTTP_200_OK)
async def reload_model():
    """Reload the production model from registry.

    Useful for updating to a newly trained model without restarting the service.
    """
    try:
        logger.info("Reloading model...")
        prediction_service.load_model()
        logger.info("Model reloaded successfully")

        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_info": prediction_service.get_model_info(),
        }

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}",
        )
