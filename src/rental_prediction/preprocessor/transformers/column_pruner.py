import pandas as pd
from loguru import logger

from rental_prediction.preprocessor.transformers.base_handler import DataHandler


class ColumnPruner(DataHandler):
    """Prune columns that are not needed for model training."""

    def __init__(self, columns_to_drop: list = None):
        """Initialize column pruner.

        Args:
            columns_to_drop: List of column names to drop. If None, uses default list.
        """
        super().__init__()
        # Default columns to drop - non-numeric or not useful for training
        self.columns_to_drop = columns_to_drop or [
            "address",
            "energy",
            "facilities",
            "zip",
            "neighborhood",
        ]

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prune specified columns from the dataframe.

        Args:
            data (pd.DataFrame): The data to process

        Returns:
            pd.DataFrame: DataFrame with specified columns removed
        """
        columns_present = [col for col in self.columns_to_drop if col in data.columns]

        if columns_present:
            logger.info(f"Pruning columns: {columns_present}")
            data = data.drop(columns=columns_present)
        else:
            logger.info("No columns to prune (already removed or not present)")

        return data
