import pandas as pd
from typing import List
from loguru import logger
from pydantic import BaseModel, Field

from rental_prediction.preprocessor.transformers.base_handler import DataHandler


class ColumnEncoderConfig(BaseModel):
    categorical_columns: List[str] = Field(
        default=["balcony", "parking", "furnished", "garage", "storage"],
        description="Columns to one-hot encode",
    )
    # Drop first category to avoid dummy variable trap (perfect multicollinearity).
    # When encoding n categories into n dummy variables, they sum to 1 (linearly dependent).
    # Critical for linear models (causes singular matrix, unstable coefficients).
    # Not required for tree-based models (Random Forest, XGBoost) but reduces dimensionality.
    drop_first: bool = Field(default=True, description="Whether to drop first category")


class ColEncoder(DataHandler):
    def __init__(self, config: ColumnEncoderConfig):
        super().__init__()
        self.config = config

    def _validate_required_cols(self, data: pd.DataFrame) -> None:
        """Validate that all required categorical columns are present in the data.

        Args:
            data (pd.DataFrame): The data to validate.

        Raises:
            ValueError: If any required column is missing.
        """
        missing_cols = set(self.config.categorical_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data by encoding categorical columns.

        Args:
            data (pd.DataFrame): The data to process.

        Raises:
            e: If an error occurs during encoding.

        Returns:
            pd.DataFrame: The encoded data.
        """
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
