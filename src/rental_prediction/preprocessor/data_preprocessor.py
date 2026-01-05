import pandas as pd
from loguru import logger

from rental_prediction.preprocessor.transformers.col_encoder import (
    ColEncoder,
    ColumnEncoderConfig,
)
from rental_prediction.preprocessor.transformers.garden_area_parser import (
    GardenAreaParser,
)
from rental_prediction.preprocessor.transformers.column_pruner import ColumnPruner


class DataPreprocessor:
    def __init__(self):
        self.handler_chain = self._create_handler_chain()

    def _create_handler_chain(self):
        """Create the preprocessing handler chain.

        Chain order:
        1. Prune unnecessary columns (address, zip, etc.)
        2. Encode categorical columns (balcony, parking, etc.)
        3. Parse garden area from text to numeric
        """
        column_pruner = ColumnPruner()
        col_encoder = ColEncoder(ColumnEncoderConfig())
        garden_parser = GardenAreaParser()

        # Set up the chain: pruner -> encoder -> parser
        column_pruner.set_next(col_encoder)
        col_encoder.set_next(garden_parser)

        return column_pruner

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting preprocessing pipeline")
        processed_df = self.handler_chain.handle(df.copy())
        return processed_df
