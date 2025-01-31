from abc import ABC, abstractmethod

import pandas as pd


class DataHandler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.process(data)
        if self._next_handler:
            # if there is a next handler, pass the data to the next handler
            return self._next_handler.handle(data)
        return data

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
