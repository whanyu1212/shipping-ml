from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: DirectoryPath
    model_name: str
    log_level: str
    rent_apart_table_name: str
