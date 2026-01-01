# Configuration Module

This module manages application configuration using Pydantic instead of Python's built-in dataclasses.

## Why Pydantic over Dataclasses?

### 1. Runtime Type Validation
Pydantic validates data types at runtime, catching configuration errors before they cause failures during training or deployment.

```python
# Pydantic - validates and coerces types
config = ModelConfig(n_trials="100")  # ✅ Auto-converts "100" → 100

# Dataclass - no validation, causes runtime errors later
config = ModelConfig(n_trials="100")  # ❌ n_trials stays as string
```

### 2. Environment Variable Loading
`BaseSettings` automatically loads configuration from environment variables and `.env` files, essential for deployment across different environments.

```python
# Automatically reads from .env file or environment
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    model_path: DirectoryPath  # Reads MODEL_PATH env var
    log_level: str             # Reads LOG_LEVEL env var
```

### 3. Advanced Validation
Built-in validators and constraints ensure configuration correctness:

```python
class ModelConfig(BaseModel):
    n_trials: int = Field(gt=0, le=1000)      # Must be 1-1000
    test_size: float = Field(gt=0.0, lt=1.0)  # Must be 0-1
```

### 4. Special Types for Configuration
Pydantic provides config-specific types like `DirectoryPath`, `FilePath`, `HttpUrl`, and `SecretStr` that validate paths exist and mask sensitive data in logs.

## Module Structure

- **`settings.py`** - Environment-specific settings (paths, log levels, database configs)
  - Uses `BaseSettings` for automatic `.env` file loading
  - Deployment configuration that varies by environment

- **`model_config.py`** - Model hyperparameters and training configuration
  - Uses `BaseModel` for type validation
  - Code-level defaults that can be overridden

## Usage

```python
from rental_prediction.config import Settings, ModelConfig

# Load environment settings
settings = Settings()  # Auto-loads from .env

# Create model configuration
model_config = ModelConfig(n_trials=200, test_size=0.25)
```

## Benefits for Production ML

1. **Fail Fast** - Configuration errors caught at startup, not mid-training
2. **Type Safety** - Prevents type-related bugs from environment variables
3. **Clear Errors** - Descriptive validation messages for misconfiguration
4. **Environment Flexibility** - Easy configuration across dev/staging/prod
5. **Documentation** - Type hints serve as inline documentation
