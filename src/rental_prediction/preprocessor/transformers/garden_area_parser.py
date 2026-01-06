import re

import pandas as pd
from loguru import logger

from rental_prediction.preprocessor.transformers.base_handler import DataHandler


class GardenAreaParser(DataHandler):
    def __init__(self):
        super().__init__()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data by parsing the garden area column.

        Args:
            data (pd.DataFrame): The data to process.

        Raises:
            e: If an error occurs during parsing.

        Returns:
            pd.DataFrame: The processed data.
        """
        logger.info("Parsing garden area")
        try:
            data["garden"] = data["garden"].apply(
                lambda x: 0 if x == "Not present" else int(re.findall(r"\d+", x)[0])
            )
            return data
        except Exception as e:
            logger.error(f"Error parsing garden area column: {e}")
            raise e
