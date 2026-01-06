# ML in Production Examples

A repository demonstrating various approaches and patterns for productionizing machine learning models, using a rental prediction use case with **automated continuous training via GitHub Actions**.

## Features

- 🏗️ **Production-ready architecture**: Modular design with data validation, preprocessing pipelines, and model registry
- 🚀 **REST API serving**: FastAPI application with automatic docs, validation, and Docker support
- ☁️ **Cloud deployment**: One-click deployment to Google Cloud Run with GitHub Actions
- 🤖 **Automated CI/CD**: GitHub Actions for testing, training, model promotion, and deployment
- 📊 **Experiment tracking**: MLflow integration for comprehensive experiment logging
- 🔄 **Continuous training**: Scheduled retraining with automatic model promotion
- 📦 **Model versioning**: Automated model registry with metadata tracking
- ✅ **Data validation**: Runtime schema validation with Pandera
- 🎯 **Hyperparameter tuning**: Optuna integration for automated optimization
- 🐳 **Containerization**: Docker and Docker Compose for easy deployment

## Getting Started

### Prerequisites
- Python 3.10+ (project uses 3.12 by default)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
# Install dependencies and create virtual environment
# This uses Python 3.12 as pinned in .python-version
uv sync

# Or specify a different Python version (3.10+)
uv sync --python 3.11

# Activate virtual environment
source .venv/bin/activate
```

**Note:** If you use a different Python version, you may need to resolve dependency conflicts. The project is tested with Python 3.12.

## Quick Start

### Local Training

```bash
# Run preprocessing demo
python examples/preprocessing_demo.py

# Train models with MLflow tracking
python examples/train_model.py

# View MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

### Model Serving (API)

```bash
# Option 1: Run with Docker Compose (recommended)
docker-compose up

# Option 2: Run locally with uvicorn
uv run uvicorn rental_prediction.api.main:app --reload

# Access API documentation
open http://localhost:8000/docs

# Make a prediction
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

### Using GitHub Actions (Continuous Training)

The repository includes automated workflows for production ML:

#### 1. Manual Training

Go to **Actions** tab → **Train Models** → **Run workflow**

Choose:
- Model type: `xgboost`, `lightgbm`, or `both`
- Number of trials for hyperparameter tuning

#### 2. Scheduled Training

Automatically runs **every Monday at 2 AM UTC** to simulate continuous training

#### 3. Automatic Model Promotion

New models are automatically promoted if they beat the baseline RMSE:
- First run: Auto-promotes (establishes baseline)
- Subsequent runs: Promotes if RMSE improves

#### 4. Cloud Deployment

Deploy to Google Cloud Run with one click:

Go to **Actions** tab → **Deploy to Cloud Run** → **Run workflow**

The API will be deployed and accessible at a public URL like:
`https://rental-prediction-api-xxxxxxxxxx-uc.a.run.app`

**Automatic deployment:** When a model is promoted, deployment is automatically triggered.

See [`.github/workflows/DEPLOYMENT.md`](.github/workflows/DEPLOYMENT.md) for complete deployment guide.

See [`.github/workflows/README.md`](.github/workflows/README.md) for detailed CI/CD documentation.

## Project Structure

```
rental_prediction/
├── .github/workflows/      # GitHub Actions CI/CD
│   ├── ci.yml             # Continuous integration
│   ├── train.yml          # Scheduled training & promotion
│   ├── deploy.yml         # Cloud Run deployment
│   ├── README.md          # CI/CD documentation
│   └── DEPLOYMENT.md      # Deployment guide
├── src/rental_prediction/
│   ├── api/               # FastAPI application
│   │   ├── main.py        # API endpoints
│   │   ├── schemas.py     # Pydantic request/response models
│   │   └── predictor.py   # Prediction service
│   ├── config/            # Pydantic configuration
│   ├── data/              # Data loading & validation
│   ├── models/            # Model implementations
│   ├── preprocessor/      # Preprocessing pipeline
│   ├── training/          # Training orchestration
│   └── utils/             # Model registry, utilities
├── examples/              # Usage examples
├── scripts/               # Production scripts
│   └── train_orchestrator.py  # Training automation
├── registry/              # Model registry metadata
│   └── production_baseline.json  # Current production model
├── models/                # Trained model artifacts
├── data/                  # Training data
├── Dockerfile             # Container definition
└── docker-compose.yml     # Multi-container setup
```

## MLOps Pipeline

### CI/CD Workflow

```mermaid
graph LR
    A[Code Push] --> B[CI Tests]
    B --> C{Tests Pass?}
    C -->|Yes| D[Schema Validation]
    C -->|No| E[❌ Fail]
    D --> F[Training Smoke Test]
    F --> G[✅ Ready]
```

### Continuous Training Pipeline

```mermaid
graph TD
    A[Scheduled/Manual Trigger] --> B[Load & Validate Data]
    B --> C[Train Models]
    C --> D[Compare vs Baseline]
    D --> E{RMSE Improved?}
    E -->|Yes| F[🚀 Promote to Production]
    E -->|No| G[⚠️ Keep Current Baseline]
    F --> H[Update registry/production_baseline.json]
    F --> I[Save Artifacts]
    G --> I
```

## Model Development

### Training a Model

```python
from rental_prediction.data import CSVLoader
from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor
from rental_prediction.models import XGBoostModel
from rental_prediction.training.trainer import Trainer
from rental_prediction.training.experiment_tracking import ExperimentTracker
from rental_prediction.config.model_config import ModelConfig

# Load data
loader = CSVLoader("data/rent_apartments.csv", validate=True)
data = loader.load()

# Setup
preprocessor = DataPreprocessor()
model = XGBoostModel(model_params={"n_estimators": 200, "max_depth": 6})
config = ModelConfig()
tracker = ExperimentTracker(experiment_name="my-experiment")

# Train
trainer = Trainer(model=model, preprocessor=preprocessor, config=config,
                 experiment_tracker=tracker)
metrics = trainer.train(data, run_name="experiment-1")

print(f"Test RMSE: {metrics['test_rmse']:.4f}")
print(f"Test R²: {metrics['test_r2']:.4f}")
```

### Hyperparameter Tuning

```python
from rental_prediction.training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(
    model_class=XGBoostModel,
    experiment_tracker=tracker
)

best_params = tuner.optimize(
    X_train, y_train, X_val, y_val,
    n_trials=100
)
```

## Configuration

The project uses Pydantic for type-safe configuration:

```python
from rental_prediction.config.model_config import ModelConfig

config = ModelConfig(
    test_size=0.2,      # Test set size
    val_size=0.2,       # Validation set size
    n_trials=100,       # Optuna trials
    random_state=42
)
```

See [`src/rental_prediction/config/README.md`](src/rental_prediction/config/README.md) for why we use Pydantic over dataclasses.

## Architecture Decisions

### Data Loading: Protocol vs ABC
We use Protocol (structural subtyping) for data loaders instead of ABC inheritance for flexibility and easier testing. See [`src/rental_prediction/data/README.md`](src/rental_prediction/data/README.md).

### Configuration: Pydantic vs Dataclasses
Pydantic provides runtime validation, environment variable loading, and type coercion essential for production. See [`src/rental_prediction/config/README.md`](src/rental_prediction/config/README.md).

### Preprocessing: Chain of Responsibility
Transformers use the Chain of Responsibility pattern for composable, testable preprocessing steps.

## API Reference

### Running the API

#### With Docker (Recommended)
```bash
# Build and start all services
docker-compose up --build

# Access services
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MLflow UI: http://localhost:5000
```

#### Local Development
```bash
# Install dependencies with uvicorn
uv sync

# Run API with hot reload
uv run uvicorn rental_prediction.api.main:app --reload --port 8000

# Access at http://localhost:8000
```

### API Endpoints

#### `GET /` - API Information
Returns API metadata and available endpoints.

#### `GET /health` - Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_name": "xgboost",
    "test_rmse": 245.67,
    "test_r2": 0.8234
  }
}
```

#### `POST /predict` - Single Prediction
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

Response:
```json
{
  "predicted_rent": 1250.50,
  "model_name": "xgboost",
  "model_version": "20240115_140530"
}
```

#### `POST /predict/batch` - Batch Predictions
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
  }'
```

#### `GET /model/info` - Model Metadata
Get information about the currently loaded model.

#### `POST /model/reload` - Reload Model
Reload the model from registry without restarting the service (useful after retraining).

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Monitoring & Observability

### MLflow Tracking

All training runs log to MLflow:
- Hyperparameters
- Metrics (RMSE, R²)
- Model artifacts
- Feature signatures

```bash
# View experiments locally
mlflow ui --backend-store-uri ./mlruns

# Or with Docker Compose
docker-compose up mlflow
# Opens at http://localhost:5000
```

### GitHub Actions Dashboard

Monitor training in the **Actions** tab:
- Training status and metrics
- Promotion decisions with reasons
- Downloadable artifacts (90-day retention)

### Production Baseline

Track the current production model in [`registry/production_baseline.json`](registry/production_baseline.json):

```json
{
  "name": "xgboost",
  "test_rmse": 245.67,
  "test_r2": 0.8234,
  "trained_at": "2024-01-15T02:00:00"
}
```

### API Monitoring

The API includes:
- Health check endpoint for load balancers
- Model reload without downtime
- Structured logging with loguru
- Docker healthcheck for container orchestration

## Customization

### Change Training Schedule

Edit [`.github/workflows/train.yml`](.github/workflows/train.yml):

```yaml
schedule:
  - cron: '0 2 * * 1'  # Every Monday at 2 AM UTC
  # Change to:
  # - cron: '0 2 * * *'    # Daily at 2 AM
  # - cron: '0 */6 * * *'  # Every 6 hours
```

### Adjust Promotion Criteria

Modify the comparison logic in `train.yml` (currently promotes if RMSE improves):

```bash
# Require 5% improvement
if (( $(echo "$NEW_RMSE < $BASELINE_RMSE * 0.95" | bc -l) )); then
  echo "should_promote=true"
fi
```

### Add Cloud Storage

Replace GitHub Actions artifacts with S3/GCS:

```yaml
- name: Upload to S3
  run: aws s3 cp models/ s3://your-bucket/models/ --recursive
```

## Roadmap

### Production Monitoring (Future Enhancement)

Complete the MLOps loop with automated monitoring via GitHub Actions:

- [ ] **Model Performance Monitoring**
  - Scheduled validation against test/validation sets
  - Track metrics over time (RMSE, R² trends)
  - Compare production predictions against actuals
  - Automatic alerts when performance degrades

- [ ] **Data Drift Detection**
  - Monitor input feature distributions
  - Statistical tests for distribution shifts
  - Schema validation for production data
  - Alert on significant drift

- [ ] **API Performance Monitoring**
  - Response time tracking
  - Error rate monitoring
  - Load testing automation
  - Uptime checks

- [ ] **Automated Alerting**
  - GitHub Issues for performance degradation
  - Slack/email notifications (optional)
  - Trigger retraining workflows automatically
  - Monitoring dashboard with GitHub Pages

- [ ] **Prediction Logging & Analysis**
  - Log predictions to database/storage
  - Analyze prediction patterns
  - A/B testing support for model versions
  - Feedback loop for ground truth labels

### Other Future Enhancements

- [ ] **Advanced Deployment**
  - Kubernetes manifests for production
  - Helm charts for easy deployment
  - Cloud deployment examples (AWS/GCP/Azure)
  - Blue-green deployment strategy

- [ ] **Model Improvements**
  - Ensemble models (stacking)
  - Neural network models
  - AutoML integration
  - Feature engineering automation

- [ ] **Data Pipeline**
  - Streaming data ingestion
  - Feature store integration
  - Real-time preprocessing
  - Data versioning with DVC

- [ ] **Enhanced Testing**
  - Integration tests for API
  - Load testing with Locust
  - Model performance regression tests
  - Contract testing for API

## Contributing

See [CLAUDE.md](CLAUDE.md) for project conventions and coding standards.

## License

[Your License Here]
