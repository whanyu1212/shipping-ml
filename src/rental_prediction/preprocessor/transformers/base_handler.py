import pandas as pd
from abc import ABC, abstractmethod


class DataHandler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler: "DataHandler") -> "DataHandler":
        """Set the next handler in the chain.

        Args:
            handler (DataHandler): The next handler to set.

        Returns:
            DataHandler: The handler that was set as next.
        """
        self._next_handler = handler
        return handler

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle the data and pass it to the next handler if exists.

        Args:
            data (pd.DataFrame): The data to handle.

        Returns:
            pd.DataFrame: The handled data.
        """
        data = self.process(data)
        if self._next_handler:
            # if there is a next handler, pass the data to the next handler
            return self._next_handler.handle(data)
        return data

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
