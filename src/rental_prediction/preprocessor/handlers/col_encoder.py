from typing import List

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .base_handler import DataHandler


class ColumnEncoderConfig(BaseModel):
    categorical_columns: List[str] = Field(
        default=["balcony", "parking", "furnished", "garage", "storage"],
        description="Columns to one-hot encode",
    )
    drop_first: bool = Field(default=True, description="Whether to drop first category")


class ColEncoder(DataHandler):
    def __init__(self, config: ColumnEncoderConfig):
        super().__init__()
        self.config = config

    def _validate_required_cols(self, data: pd.DataFrame):
        missing_cols = set(self.config.categorical_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding categorical columns")
        self._validate_required_cols(data)
        try:
            encoded_data = pd.get_dummies(
                data,
                columns=self.config.categorical_columns,
                drop_first=self.config.drop_first,
            )
            return encoded_data
        except Exception as e:
            logger.error(f"Error encoding columns: {e}")
            raise e
