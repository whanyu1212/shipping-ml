import pandas as pd
from loguru import logger

from .handlers.col_encoder import ColEncoder, ColumnEncoderConfig
from .handlers.garden_area_parser import GardenAreaParser


class DataPreprocessor:
    def __init__(self):
        self.handler_chain = self._create_handler_chain()

    def _create_handler_chain(self):
        col_encoder = ColEncoder(ColumnEncoderConfig())
        garden_parser = GardenAreaParser()

        col_encoder.set_next(garden_parser)
        return col_encoder

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting preprocessing pipeline")
        processed_df = self.handler_chain.handle(df.copy())
        return processed_df
