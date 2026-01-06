# Rental Prediction API

FastAPI-based REST API for serving rental price predictions in production. Provides a production-ready interface to the trained ML models with automatic validation, documentation, and model reloading capabilities.

## Features

- 🚀 **Production-ready**: FastAPI with async support, CORS, and lifecycle management
- ✅ **Request validation**: Pydantic schemas with automatic type checking
- 📊 **Model serving**: Loads models from registry, supports hot-reloading
- 📝 **Auto-documentation**: Interactive Swagger UI and ReDoc
- 🏥 **Health checks**: Docker-compatible health endpoint
- 🔄 **Zero-downtime updates**: Reload model without restarting service
- 📦 **Batch predictions**: Efficient batch processing (up to 1000 apartments)

## Architecture

```
api/
├── main.py           # FastAPI app with endpoints
├── schemas.py        # Pydantic request/response models
├── predictor.py      # Prediction service (model loading & inference)
└── __init__.py
```

### Components

**main.py** - FastAPI application
- Endpoint definitions and routing
- Request/response handling
- Error handling and logging
- CORS middleware
- Lifecycle management (model loading on startup)

**schemas.py** - Data validation
- Pydantic models for type-safe API contracts
- Input validation (ranges, types, constraints)
- Response serialization
- Example payloads for documentation

**predictor.py** - Model inference service
- Loads production model from `registry/production_baseline.json`
- Handles preprocessing pipeline
- Single and batch predictions
- Model metadata management

## Running the API

### Local Development

```bash
# Install dependencies
uv sync

# Run with hot reload
uv run uvicorn rental_prediction.api.main:app --reload

# Or specify host/port
uv run uvicorn rental_prediction.api.main:app --host 0.0.0.0 --port 8000
```

### With Docker

```bash
# Build and run with Docker Compose (recommended)
docker-compose up

# Or build manually
docker build -t rental-api .
docker run -p 8000:8000 rental-api
```

### Access Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000

## API Endpoints

### `GET /` - API Information

Returns basic API metadata and available endpoints.

**Response:**
```json
{
  "name": "Rental Prediction API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

### `GET /health` - Health Check

Health check endpoint for load balancers and container orchestration.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_name": "xgboost",
    "model_class": "XGBoostModel",
    "test_rmse": 245.67,
    "test_r2": 0.8234,
    "trained_at": "2024-01-15T14:05:30"
  }
}
```

**Status Codes:**
- `200 OK` - Service is healthy and model is loaded
- Returns `"unhealthy"` status if model failed to load (still returns 200)

**Example:**
```bash
curl http://localhost:8000/health
```

**Use cases:**
- Docker HEALTHCHECK
- Kubernetes liveness/readiness probes
- Load balancer health checks

---

### `GET /model/info` - Model Information

Get detailed information about the currently loaded production model.

**Response:**
```json
{
  "model_name": "xgboost",
  "model_class": "XGBoostModel",
  "version": "20240115_140530",
  "test_rmse": 245.67,
  "test_r2": 0.8234,
  "trained_at": "2024-01-15T14:05:30",
  "available_features": [
    "area", "rooms", "construction_year", "balcony",
    "parking", "furnished", "garage", "storage", "garden"
  ]
}
```

**Status Codes:**
- `200 OK` - Model info returned successfully
- `503 Service Unavailable` - Model not loaded

**Example:**
```bash
curl http://localhost:8000/model/info
```

---

### `POST /predict` - Single Prediction

Make a rental price prediction for a single apartment.

**Request Body:**
```json
{
  "features": {
    "area": 85.5,
    "rooms": 3,
    "construction_year": 2010,
    "balcony": "yes",
    "parking": "yes",
    "furnished": "no",
    "garage": "no",
    "storage": "yes",
    "garden": "50 m²"
  }
}
```

**Response:**
```json
{
  "predicted_rent": 1250.50,
  "model_name": "xgboost",
  "model_version": "20240115_140530"
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `422 Unprocessable Entity` - Invalid input (validation error)
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Prediction failed

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "area": 85.5,
      "rooms": 3,
      "construction_year": 2010,
      "balcony": "yes",
      "parking": "yes",
      "furnished": "no",
      "garage": "no",
      "storage": "yes",
      "garden": "50 m²"
    }
  }'
```

---

### `POST /predict/batch` - Batch Predictions

Make predictions for multiple apartments in a single request (up to 1000).

**Request Body:**
```json
{
  "apartments": [
    {
      "area": 85.5,
      "rooms": 3,
      "construction_year": 2010,
      "balcony": "yes",
      "parking": "yes",
      "furnished": "no",
      "garage": "no",
      "storage": "yes",
      "garden": "50 m²"
    },
    {
      "area": 120.0,
      "rooms": 4,
      "construction_year": 2015,
      "balcony": "yes",
      "parking": "yes",
      "furnished": "yes",
      "garage": "yes",
      "storage": "yes",
      "garden": "100 m²"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [1250.50, 1850.75],
  "model_name": "xgboost",
  "model_version": "20240115_140530",
  "count": 2
}
```

**Constraints:**
- Minimum: 1 apartment
- Maximum: 1000 apartments per request

**Status Codes:**
- `200 OK` - Predictions successful
- `422 Unprocessable Entity` - Invalid input or too many apartments
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Prediction failed

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "apartments": [
      {
        "area": 85.5,
        "rooms": 3,
        "construction_year": 2010,
        "balcony": "yes",
        "parking": "yes",
        "furnished": "no",
        "garage": "no",
        "storage": "yes",
        "garden": "50 m²"
      }
    ]
  }'
```

**Use cases:**
- Bulk price estimation for property portfolios
- Integration with data pipelines
- Batch processing during off-peak hours

---

### `POST /model/reload` - Reload Model

Reload the production model from registry without restarting the service.

**Response:**
```json
{
  "status": "success",
  "message": "Model reloaded successfully",
  "model_info": {
    "model_name": "xgboost",
    "test_rmse": 245.67,
    "test_r2": 0.8234,
    "trained_at": "2024-01-15T14:05:30"
  }
}
```

**Status Codes:**
- `200 OK` - Model reloaded successfully
- `500 Internal Server Error` - Reload failed

**Example:**
```bash
curl -X POST "http://localhost:8000/model/reload"
```

**Use cases:**
- Deploy newly trained model without downtime
- Recover from model loading failures
- Switch between model versions

**Workflow integration:**
```bash
# After GitHub Actions promotes a new model
git pull  # Get updated registry/production_baseline.json
curl -X POST http://localhost:8000/model/reload
```

## Input Validation

All inputs are validated using Pydantic schemas:

### ApartmentFeatures Constraints

| Field | Type | Constraints | Example |
|-------|------|-------------|---------|
| `area` | float | > 0 | `85.5` |
| `rooms` | int | 1-10 | `3` |
| `construction_year` | int | >= 0 | `2010` |
| `balcony` | string | Any string | `"yes"` |
| `parking` | string | Any string | `"yes"` |
| `furnished` | string | Any string | `"no"` |
| `garage` | string | Any string | `"no"` |
| `storage` | string | Any string | `"yes"` |
| `garden` | string | Any string | `"50 m²"` |

**Validation errors** return `422 Unprocessable Entity` with detailed error messages:

```json
{
  "detail": [
    {
      "loc": ["body", "features", "area"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

## Error Handling

The API uses standard HTTP status codes:

| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Request successful |
| 422 | Unprocessable Entity | Validation error (invalid input) |
| 500 | Internal Server Error | Prediction failed or server error |
| 503 | Service Unavailable | Model not loaded |

All errors return structured JSON:

```json
{
  "detail": "Error message explaining what went wrong"
}
```

## Model Loading

The API automatically loads the production model on startup using this flow:

1. **Startup**: `lifespan` manager loads model from `registry/production_baseline.json`
2. **Reads baseline**: Gets model name and metadata
3. **Loads model**: Retrieves model pickle from `models/{model_name}/latest/`
4. **Ready**: Service starts accepting prediction requests

If model loading fails on startup:
- API still starts (degrades gracefully)
- Health endpoint returns `"unhealthy"`
- Prediction endpoints return `503 Service Unavailable`
- Use `/model/reload` to retry loading

## CORS Configuration

CORS is configured to allow all origins for development:

```python
allow_origins=["*"]  # Configure appropriately for production
```

**For production**, restrict to specific origins:

```python
allow_origins=[
    "https://yourdomain.com",
    "https://app.yourdomain.com",
]
```

## Testing the API

### Manual Testing

Use the interactive Swagger UI at http://localhost:8000/docs:
1. Expand an endpoint
2. Click "Try it out"
3. Fill in example values
4. Click "Execute"
5. View response

### Integration Testing

```python
from fastapi.testclient import TestClient
from rental_prediction.api.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "features": {
            "area": 85.5,
            "rooms": 3,
            "construction_year": 2010,
            "balcony": "yes",
            "parking": "yes",
            "furnished": "no",
            "garage": "no",
            "storage": "yes",
            "garden": "50 m²"
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rent" in data
    assert data["predicted_rent"] > 0
```

### Load Testing

```bash
# Install vegeta (load testing tool)
brew install vegeta  # macOS

# Create request file
cat > request.txt <<EOF
POST http://localhost:8000/predict
Content-Type: application/json
@predict_payload.json
EOF

# Run load test
echo "GET http://localhost:8000/health" | \
  vegeta attack -duration=30s -rate=50 | \
  vegeta report
```

## Deployment Considerations

### Environment Variables

Configure via environment variables (optional):

```bash
# Model registry path (default: "models")
MODEL_REGISTRY_PATH=models

# Use production baseline (default: true)
USE_PRODUCTION_BASELINE=true

# Log level (default: INFO)
LOG_LEVEL=INFO
```

### Production Checklist

- [ ] Configure CORS to allow only specific origins
- [ ] Set up HTTPS/TLS termination (e.g., nginx reverse proxy)
- [ ] Configure logging to external system (e.g., CloudWatch, Datadog)
- [ ] Set up monitoring and alerting
- [ ] Implement rate limiting (e.g., with slowapi)
- [ ] Add authentication/API keys if needed
- [ ] Configure autoscaling based on CPU/memory
- [ ] Set resource limits in Docker/Kubernetes

### Docker Healthcheck

The Dockerfile includes a healthcheck:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

This ensures:
- Container orchestrators can detect unhealthy instances
- Automatic restarts on repeated failures
- Load balancers route traffic only to healthy instances

## API Performance

### Latency Benchmarks

Typical response times (local, single apartment):
- `/health`: < 5ms
- `/model/info`: < 5ms
- `/predict`: 10-20ms (includes preprocessing)
- `/predict/batch` (100 items): 50-100ms

### Optimization Tips

1. **Batch predictions**: Use `/predict/batch` for multiple apartments (5-10x faster than multiple single requests)
2. **Model caching**: Model stays in memory after loading (no reload overhead)
3. **Async processing**: FastAPI handles requests concurrently
4. **Preprocessing**: One-time preprocessing per batch, not per item

## Monitoring

### Recommended Metrics

- **Request rate**: Requests per second
- **Response time**: p50, p95, p99 latency
- **Error rate**: 4xx and 5xx responses
- **Model version**: Track which model is in production
- **Memory usage**: Ensure model fits in memory

### Logging

The API uses `loguru` for structured logging:

```python
logger.info("Model loaded successfully")
logger.error(f"Prediction failed: {e}")
```

**Production logging** should send to:
- CloudWatch Logs (AWS)
- Stackdriver (GCP)
- Application Insights (Azure)
- Datadog, New Relic, etc.

## Troubleshooting

### Model not loading on startup

**Error**: `Failed to load model on startup`

**Solutions**:
1. Check `registry/production_baseline.json` exists
2. Verify model file exists at path specified in baseline
3. Check file permissions
4. Ensure all dependencies are installed
5. Use `/model/reload` to retry

### Validation errors on prediction

**Error**: `422 Unprocessable Entity`

**Solutions**:
1. Check request matches schema (see Input Validation section)
2. Ensure all required fields are present
3. Verify field types (int vs float vs string)
4. Check value ranges (e.g., `area > 0`, `rooms` between 1-10)

### Container fails healthcheck

**Error**: Docker container marked unhealthy

**Solutions**:
1. Check model loaded successfully (logs)
2. Verify port 8000 is accessible inside container
3. Ensure `requests` package is installed
4. Increase healthcheck timeout/start-period

## Future Enhancements

- [ ] **Authentication**: API key or JWT-based auth
- [ ] **Rate limiting**: Prevent abuse with request quotas
- [ ] **Caching**: Redis cache for repeated predictions
- [ ] **Async predictions**: Background task queue for large batches
- [ ] **Model A/B testing**: Serve multiple models simultaneously
- [ ] **Explainability**: SHAP values for feature importance
- [ ] **Monitoring dashboard**: Real-time metrics and logs
- [ ] **Versioned API**: `/v1/predict`, `/v2/predict` for breaking changes

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
