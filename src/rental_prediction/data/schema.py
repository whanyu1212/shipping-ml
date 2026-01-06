"""Data schema validation using Pandera.

This module defines the schema for rental apartment data, ensuring
data quality and consistency across the pipeline.

References:
- Pandera DataFrameModel: https://pandera.readthedocs.io/en/latest/dataframe_models.html
- Best Practices: https://khuyentran1401.github.io/reproducible-data-science/testing_data/pandera.html
"""

import datetime

import pandera.pandas as pa
from pandera.typing.pandas import Series


class DfSchema(pa.DataFrameModel):
    """Rental apartment data schema v1.0.

    Validates raw rental data from CSV/BigQuery sources before processing.
    Uses Pandera's class-based API for type-safe data validation.
    """

    # Basic field definitions
    address: Series[str] = pa.Field(nullable=False)
    area: Series[float] = pa.Field(gt=0, nullable=False)
    construction_year: Series[int] = pa.Field(ge=0, coerce=True)
    rooms: Series[int] = pa.Field(gt=0, nullable=False)
    bedrooms: Series[int] = pa.Field(gt=0, nullable=False)
    bathrooms: Series[int] = pa.Field(gt=0, nullable=False)
    balcony: Series[str] = pa.Field(isin=["yes", "no"], nullable=False)
    storage: Series[str] = pa.Field(isin=["yes", "no"], nullable=False)
    parking: Series[str] = pa.Field(isin=["yes", "no"], nullable=False)
    furnished: Series[str] = pa.Field(isin=["yes", "no"], nullable=False)
    garage: Series[str] = pa.Field(isin=["yes", "no"], nullable=False)
    garden: Series[str] = pa.Field(nullable=False)
    energy: Series[str] = pa.Field(nullable=True)
    facilities: Series[str] = pa.Field(nullable=True)
    zip: Series[str] = pa.Field(nullable=False)
    neighborhood: Series[str] = pa.Field(nullable=False)
    rent: Series[int] = pa.Field(gt=0, nullable=False, coerce=True)

    class Config:
        """Pandera validation configuration.

        - strict: Reject DataFrames with extra columns not in schema
        - coerce: Automatically convert types to match schema (with field-level coerce)
        """

        name = "RentalDataSchema"
        strict = True
        coerce = True

    # Custom checks ----------------------------------------------------------
    @pa.check("garden", error="Must start with 'Present' or 'Not present'")
    def check_garden(cls, series: Series[str]) -> Series[bool]:
        """Validate garden field format."""
        return series.str.startswith(("Present", "Not present"))

    @pa.check("energy", error="Energy rating must start with A-G")
    def check_energy(cls, series: Series[str]) -> Series[bool]:
        """Validate energy rating format (A-G)."""
        return series.isna() | series.str.startswith(tuple("ABCDEFG"))

    @pa.check("zip", error="Zip code must start with numeric character")
    def check_zip(cls, series: Series[str]) -> Series[bool]:
        """Validate zip code format."""
        return series.str.match(r"^\d")

    @pa.check("construction_year", error="Year must be between 0 and next year")
    def check_construction_year(cls, series: Series[int]) -> Series[bool]:
        """Check if construction year is within reasonable range.

        Allows historic buildings from 0 onwards (common in European cities).
        """
        current_year = datetime.datetime.now().year
        return (series >= 0) & (series <= current_year + 1)
