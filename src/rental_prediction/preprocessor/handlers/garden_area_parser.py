import re

import pandas as pd
from loguru import logger

from .base_handler import DataHandler


class GardenAreaParser(DataHandler):
    def __init__(self):
        super().__init__()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Parsing garden area")
        try:
            data["garden"] = data["garden"].apply(
                lambda x: 0 if x == "Not present" else int(re.findall(r"\d+", x)[0])
            )
            return data
        except Exception as e:
            logger.error(f"Error parsing garden area column: {e}")
            raise e
