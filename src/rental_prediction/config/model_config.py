from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for model training and hyperparameter tuning."""

    model_name: str = "rf"
    n_trials: int = 100
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
