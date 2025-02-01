from pydantic import BaseModel, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: DirectoryPath
    model_name: str
    log_level: str
    rent_apart_table_name: str


class ModelConfig(BaseModel):
    model_name: str = "rf"
    n_trials: int = 100
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
