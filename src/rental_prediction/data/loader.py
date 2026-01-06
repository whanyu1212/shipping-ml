"""Data loading with Protocol-based interface.

This module provides flexible data loading using structural subtyping (Protocol)
instead of inheritance, allowing any class with a load() method to work as a loader.

Protocol vs ABC:
- Protocol: Duck typing with type checking, no inheritance required
- ABC: Requires inheritance, more rigid but enforces implementation

For data loaders, Protocol is preferred for flexibility and ease of testing.
"""

import pandas as pd
from typing import Protocol, runtime_checkable, Optional
from pathlib import Path
import pandera.pandas as pa

from rental_prediction.data.schema import DfSchema


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""

    pass


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for data loaders using structural subtyping.

    Any class implementing a load() -> pd.DataFrame method is compatible.
    No inheritance required - enables duck typing with type safety.
    """

    def load(self) -> pd.DataFrame:
        """Load data and return as DataFrame.

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            DataLoadError: If loading fails
        """
        ...


class CSVLoader:
    """CSV data loader with validation and caching.

    Example:
        >>> loader = CSVLoader("data.csv", validate=True)
        >>> df = loader.load()
    """

    def __init__(
        self,
        file_path: str | Path,
        validate: bool = True,
        cache: bool = False,
    ):
        """Initialize CSV loader.

        Args:
            file_path (str | Path): Path to CSV file
            validate (bool): Whether to validate data with DfSchema
            cache (bool): Whether to cache loaded data in memory
        """
        self.file_path = Path(file_path)
        self.validate = validate
        self.cache = cache
        self._cached_data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load data from CSV with optional validation and caching.

        Returns:
            pd.DataFrame: Validated (if enabled) DataFrame

        Raises:
            DataLoadError: If file not found or validation fails
        """
        # Return cached data if available
        if self.cache and self._cached_data is not None:
            return self._cached_data

        try:
            df = pd.read_csv(self.file_path)

            # Handle legacy column names for backwards compatibility
            legacy_column_mapping = {
                "constraction_year": "construction_year",  # Fix typo in source data
            }
            df = df.rename(columns=legacy_column_mapping)

            if self.validate:
                # Validate with schema, using lazy=True for production resilience
                df = DfSchema.validate(df, lazy=True)

            if self.cache:
                self._cached_data = df

            return df

        except FileNotFoundError as e:
            raise DataLoadError(f"CSV file not found: {self.file_path}") from e
        except pa.errors.SchemaError as e:
            raise DataLoadError(f"Schema validation failed: {e}") from e


# This is not implemented yet
class BigQueryLoader:
    """BigQuery data loader with validation.

    Example:
        >>> loader = BigQueryLoader("project.dataset.table", credentials_path="key.json")
        >>> df = loader.load()
    """

    def __init__(
        self,
        table_id: str,
        credentials_path: Optional[str | Path] = None,
        validate: bool = True,
    ):
        """Initialize BigQuery loader.

        Args:
            table_id (str): Fully qualified table ID (project.dataset.table)
            credentials_path (Optional[str | Path]): Path to service account credentials JSON
            validate (bool): Whether to validate data with DfSchema
        """
        self.table_id = table_id
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self.validate = validate

    def load(self) -> pd.DataFrame:
        """Load data from BigQuery table.

        Returns:
            pd.DataFrame: Validated (if enabled) DataFrame

        Raises:
            DataLoadError: If connection or validation fails
        """
        # TODO: Implement BigQuery loading logic
        # from google.cloud import bigquery
        # from google.oauth2 import service_account
        #
        # if self.credentials_path:
        #     credentials = service_account.Credentials.from_service_account_file(
        #         self.credentials_path
        #     )
        #     client = bigquery.Client(credentials=credentials)
        # else:
        #     client = bigquery.Client()
        #
        # query = f"SELECT * FROM `{self.table_id}`"
        # df = client.query(query).to_dataframe()
        #
        # if self.validate:
        #     df = DfSchema.validate(df, lazy=True)
        #
        # return df
        raise NotImplementedError("BigQuery loader implementation pending")
