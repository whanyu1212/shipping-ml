# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository demonstrates production ML patterns using a rental prediction use case. It showcases the full ML lifecycle: data ingestion → preprocessing → training → experiment tracking → model registry → API serving → continuous training via GitHub Actions.

**Key focus areas:**
- Production-ready code patterns (Protocol interfaces, Pydantic validation, Chain of Responsibility)
- Automated continuous training with model promotion
- REST API serving with FastAPI
- **Cloud deployment**: One-click deployment to Google Cloud Run
- MLflow experiment tracking and model registry
- Docker containerization and GitHub Actions CI/CD
- **Complete MLOps pipeline**: Train → Promote → Deploy automatically

## Development Setup

```bash
# Install uv (if not already installed)
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# Python version is pinned to 3.12 (see .python-version)
uv sync                     # Install all dependencies, create .venv
source .venv/bin/activate   # Activate virtual environment

# Verify installation
uv run python -c "import rental_prediction; print('✓ Package installed')"
```

## Common Commands

### Data & Preprocessing
```bash
# Load and validate data with Pandera schema
python examples/load_csv_data.py

# Demo preprocessing pipeline (Chain of Responsibility pattern)
python examples/preprocessing_demo.py
```

### Training Models
```bash
# Train a single model (no MLflow tracking)
python examples/train_model.py

# Production training orchestrator (used by GitHub Actions)
python scripts/train_orchestrator.py \
  --model-type both \
  --n-trials 50 \
  --output artifacts/training_report.json

# Options:
#   --model-type: xgboost | lightgbm | both
#   --n-trials: Number of Optuna hyperparameter tuning trials
#   --data-path: Path to CSV data (default: data/rent_apartments.csv)
```

### Experiment Tracking
```bash
# Start MLflow UI (view all experiments)
mlflow ui --backend-store-uri ./mlruns
# Opens at http://localhost:5000

# Or use Optuna Dashboard for hyperparameter optimization viz
optuna-dashboard sqlite:///optuna.db
```

### API Serving
```bash
# Run API locally with hot reload
uv run uvicorn rental_prediction.api.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# Run with Docker Compose (recommended for production)
docker-compose up
# Includes both API (port 8000) and MLflow UI (port 5000)

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"area": 85.5, "rooms": 3, "construction_year": 2010, "balcony": "yes", "parking": "yes", "furnished": "no", "garage": "no", "storage": "yes", "garden": "50 m²"}}'

# Reload model without restarting (after new training)
curl -X POST "http://localhost:8000/model/reload"
```

### Cloud Deployment (Google Cloud Run)
```bash
# Deploy to production (manual trigger)
# Go to GitHub Actions → "Deploy to Cloud Run" → Run workflow

# The deployment is also automatic:
# When a model is promoted → commits to main → auto-deploys to Cloud Run

# Your API will be accessible at:
# https://rental-prediction-api-xxxxx-uc.a.run.app

# Test deployed API
export API_URL="https://rental-prediction-api-xxxxx-uc.a.run.app"
curl $API_URL/health

# Configuration
PROJECT_ID: fleet-anagram-244304
REGION: us-central1
SERVICE_NAME: rental-prediction-api
# See .github/workflows/deploy.yml for full config
```

**Key files for deployment:**
- `.github/workflows/deploy.yml` - Cloud Run deployment automation
- `.github/workflows/DEPLOYMENT.md` - Complete deployment guide
- `Dockerfile` - Includes system deps (libgomp1 for LightGBM)
- `GCP_SA_KEY` GitHub Secret - Service account authentication

### Code Quality
```bash
# No automated test suite yet - run smoke tests manually
uv run python -c "
from rental_prediction.data import CSVLoader
loader = CSVLoader('data/rent_apartments.csv', validate=True)
data = loader.load()
print(f'✓ Schema validation passed: {len(data)} records')
"

# Formatting and linting (optional - not enforced in CI)
black src/ examples/ scripts/
isort src/ examples/ scripts/
flake8 src/ examples/ scripts/
```

## Architecture & Design Patterns

### 1. Protocol-Based Interfaces (Structural Subtyping)

**Why Protocol over ABC**: Flexibility for third-party integrations and easier testing.

```python
# rental_prediction/data/loader.py
class DataLoader(Protocol):
    def load(self) -> pd.DataFrame: ...

# Any class with load() works - no inheritance needed
class CSVLoader:  # Implicitly satisfies DataLoader
    def load(self) -> pd.DataFrame: ...
```

See `src/rental_prediction/data/README.md` for detailed rationale.

### 2. Pydantic for Configuration (Not Dataclasses)

**Why Pydantic**: Runtime validation, environment variable loading, type coercion essential for production.

```python
# rental_prediction/config/model_config.py
class ModelConfig(BaseModel):
    model_name: Literal["xgboost", "lightgbm"]  # Type-safe constraints
    n_trials: int = Field(gt=0, le=1000)        # Runtime validation
    test_size: float = Field(gt=0.0, lt=1.0)
```

See `src/rental_prediction/config/README.md` for detailed comparison with dataclasses.

### 3. Chain of Responsibility (Preprocessing)

Transformers are independent, composable, and testable:

```python
# rental_prediction/preprocessor/transformers/
ConstructionYearTransformer → BinaryFeaturesTransformer →
CategoricalTransformer → NumericScaler

# Each transformer can be tested in isolation
# Order matters - defined in DataPreprocessor
```

### 4. Classmethod Pattern for Hyperparameter Search

**Critical pattern**: `suggest_params()` is a `@classmethod` because it's called BEFORE instantiation during Optuna trials.

```python
# rental_prediction/models/base_model.py
@classmethod
def suggest_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
    """Called on MODEL CLASS, not instance"""

# rental_prediction/training/hyperparameter_tuning.py
# Step 1: Get params from class (before any instance exists)
params = XGBoostModel.suggest_params(trial)
# Step 2: Now create instance with suggested params
model = XGBoostModel(model_params=params)
```

See inline comments in `src/rental_prediction/training/hyperparameter_tuning.py` for detailed explanation.

### 5. Data Validation with Pandera

**Always validate raw data** before processing. Use `pandera.pandas` import (not `pandera`).

```python
# rental_prediction/data/schema.py
class DfSchema(pa.DataFrameModel):
    area: float = pa.Field(gt=0)
    rooms: int = pa.Field(ge=1, le=10)
    construction_year: int = pa.Field(ge=0)  # Historic buildings from 1600s exist
    # ... validates + coerces types automatically

# CSVLoader validates on load
loader = CSVLoader("data.csv", validate=True)  # Fails fast on bad data
```

### 6. Model Registry Pattern

Models are versioned by timestamp in `models/{model_name}/{timestamp}/`:

```python
# rental_prediction/utils/model_registry.py
registry = ModelRegistry(registry_path=Path("models"))
registry.save_model(model, "xgboost", metadata={...})
# Saves to: models/xgboost/20240115_140530/model.pkl

# Production model tracked in registry/production_baseline.json
# Updated by GitHub Actions on successful promotion
```

## Continuous Training & Deployment Pipeline (GitHub Actions)

### Automated Workflow (Complete MLOps Loop)

1. **Trigger**: Every Monday at 2 AM UTC (or manual via Actions tab)
2. **Train**: Runs `scripts/train_orchestrator.py` with both models
3. **Compare**: New model test_rmse vs `registry/production_baseline.json`
4. **Promote**: If `new_rmse < baseline_rmse`, update baseline and commit to `main`
5. **Deploy**: Push to `main` triggers Cloud Run deployment automatically
6. **Artifacts**: Saves models, MLflow runs, training_report.json (90 days)

**Result**: New models automatically deployed to production with zero manual intervention!

### Key Files

- `.github/workflows/train.yml` - Training automation with promotion logic
- `.github/workflows/deploy.yml` - Cloud Run deployment automation
- `.github/workflows/ci.yml` - Schema validation and smoke tests
- `registry/production_baseline.json` - **Source of truth** for production model
- `scripts/train_orchestrator.py` - Coordinates training, comparison, reporting
- `Dockerfile` - Multi-stage build with ML library dependencies (libgomp1)

### Production Baseline

**Critical**: `registry/production_baseline.json` determines which model the API loads.

```json
{
  "name": "xgboost",
  "test_rmse": 245.67,
  "test_r2": 0.8234,
  "trained_at": "2024-01-15T14:05:30"
}
```

**First run**: Placeholder (test_rmse: 9999.0) ensures first model is auto-promoted.
**Subsequent runs**: Only promotes if metrics improve.
**API**: Loads model specified in baseline via `PredictionService.load_model()`

### Manual Training Trigger

Go to **Actions** → **Train Models** → **Run workflow**:
- Choose model type: `xgboost`, `lightgbm`, or `both`
- Set n_trials for hyperparameter tuning (default: 50)

## Project Structure (Key Components)

```
rental_prediction/
├── config/              # Pydantic-based configuration
│   ├── model_config.py  # Training hyperparameters, constraints
│   └── settings.py      # Environment settings (paths, log levels)
├── data/                # Protocol interfaces + Pandera validation
│   ├── loader.py        # DataLoader Protocol (not ABC)
│   ├── schema.py        # DfSchema for validation
│   └── ingestion/       # BigQuery, CSV loaders
├── models/              # Model implementations (XGBoost, LightGBM)
│   ├── base_model.py    # BaseModel with suggest_params() classmethod
│   ├── xgboost_model.py
│   └── lightgbm_model.py
├── preprocessor/        # Chain of Responsibility pipeline
│   ├── data_preprocessor.py      # Orchestrates transformer chain
│   └── transformers/             # Individual transformers
├── training/            # Training orchestration
│   ├── trainer.py                # 2-stage train/val/test splitting
│   ├── hyperparameter_tuning.py  # Optuna integration
│   └── experiment_tracking.py    # MLflow wrapper
├── api/                 # FastAPI serving
│   ├── main.py          # Endpoints + lifecycle management
│   ├── schemas.py       # Pydantic request/response models
│   └── predictor.py     # Model loading + inference
└── utils/
    └── model_registry.py         # Versioned model storage

scripts/
└── train_orchestrator.py         # Production training script

registry/
└── production_baseline.json      # Current production model

models/                            # Model artifacts (gitignored)
└── {ModelName}/{timestamp}/
    ├── model.pkl
    └── metadata.json

.github/workflows/
├── train.yml          # Continuous training + promotion
├── deploy.yml         # Cloud Run deployment
├── ci.yml             # Schema validation + smoke tests
├── README.md          # CI/CD documentation
└── DEPLOYMENT.md      # Cloud deployment guide
```

## Important Data Quirks

- **Historic buildings**: `construction_year` can be as early as 1600s (constraint: `>= 0`, not `>= 1900`)
- **Data quality issue**: Some `construction_year` values are `1005` (appears to be data entry error)
- **Legacy column name**: Original dataset has typo `constraction_year` → CSVLoader maps to `construction_year`

## Git Workflow

- **Main branch**: `main` (use for PRs)
- **Development branch**: `develop`
- **Commit message suffix**: `🤖 Generated with Claude Code` (for commits by Claude)
- **Important**: Commit `registry/production_baseline.json` - required for GitHub Actions workflow

## Environment & Dependencies

- **Python**: 3.10+ (project uses 3.12 by default, pinned in `.python-version`)
- **Package manager**: `uv` (NOT Poetry - migrated from Poetry to uv)
- **Key libraries**:
  - ML: scikit-learn, xgboost, lightgbm, optuna
  - Tracking: MLflow, optuna-dashboard
  - Validation: Pandera (use `pandera.pandas`, not `pandera`)
  - Config: Pydantic + pydantic-settings
  - API: FastAPI, uvicorn
  - Data: pandas, numpy
  - BigQuery: google-cloud-bigquery (optional)

## Coding Conventions

### Style
- Type hints required for all function signatures
- PEP 8 with 100-character line length
- Google-style docstrings (autodocstring format)
- No single-letter variables except loop indices

### ML-Specific
- Always validate data with Pandera before processing
- Use `lazy=True` for production validation (collect all errors)
- Log all experiments to MLflow with params, metrics, artifacts
- Document hyperparameter choices in model classes

### File Organization
- **Library code**: No `__main__` blocks in `src/rental_prediction/`
- **Examples**: Demonstration scripts in `examples/`
- **Scripts**: Production CLI tools in `scripts/` with argparse
- **Tests**: Currently minimal - `tests/` exists but not comprehensive

## Package Management with uv

```bash
# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update dependencies
uv sync

# Run commands in venv
uv run python script.py
uv run pytest tests/
```

## Docker & Deployment

```bash
# Build API container
docker build -t rental-api .

# Run with Docker Compose (API + MLflow UI)
docker-compose up

# Container healthcheck uses localhost (correct - runs inside container)
# Dockerfile includes: requests.get('http://localhost:8000/health')
```

**Important Dockerfile requirements:**
1. **Multi-stage build**: Builder stage needs `src/` and `README.md` for hatchling
2. **System dependencies**: `libgomp1` required for LightGBM/XGBoost (OpenMP library)
3. **Registry folder**: Must include `registry/` with `production_baseline.json`
4. **Models folder**: Include `models/` directory (can be empty with `.gitkeep`)

```dockerfile
# Builder stage - needs src/ and README.md
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ /app/src/
RUN uv sync --frozen --no-dev

# Final stage - needs libgomp1 for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

## API Endpoints Summary

- `GET /` - API info
- `GET /health` - Health check (Docker HEALTHCHECK compatible)
- `GET /model/info` - Current model metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions (max 1000)
- `POST /model/reload` - Hot-reload model without restart

See `src/rental_prediction/api/README.md` for comprehensive API documentation.

## Troubleshooting

### Import Error: "No module named 'rental_prediction'"
Ensure package is installed: `uv sync` then `source .venv/bin/activate`

### Pandera Import Error
Use `import pandera.pandas as pa` (NOT `import pandera as pa`)

### Model Not Loading in API
1. Check `registry/production_baseline.json` exists
2. Verify model file exists at path in baseline
3. Train a model first: `python scripts/train_orchestrator.py --model-type xgboost --n-trials 10`

### GitHub Actions Promotion Not Working
1. Ensure `registry/production_baseline.json` is committed to repo
2. Check workflow logs for `jq` parsing errors
3. Verify training_report.json has `best_model` key

## References

Key README files for deep dives:
- `README.md` - Project overview, API usage, CI/CD pipeline
- `src/rental_prediction/config/README.md` - Why Pydantic over dataclasses
- `src/rental_prediction/data/README.md` - Why Protocol over ABC
- `src/rental_prediction/api/README.md` - Complete API documentation
- `.github/workflows/README.md` - CI/CD pipeline details
