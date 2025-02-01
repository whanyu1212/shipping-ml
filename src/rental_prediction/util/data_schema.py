import datetime

import pandera as pa
from pandera.typing import Series


class DfSchema(pa.DataFrameModel):
    # Basic field definitions
    address: Series[str] = pa.Field(nullable=False)
    area: Series[float] = pa.Field(gt=0, nullable=False)
    constraction_year: Series[int] = pa.Field(gt=0)
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
    rent: Series[int] = pa.Field(gt=0, nullable=False)

    # Custom checks ----------------------------------------------------------
    @pa.check("garden", error="Must start with 'Present' or 'Not Present'")
    def check_garden(cls, series: Series[str]) -> Series[bool]:
        return series.str.startswith(("Present", "Not present"))

    @pa.check("energy", error="Must start with A-G")
    def check_energy(cls, series: Series[str]) -> Series[bool]:
        return series.isna() | series.str.startswith(tuple("ABCDEFG"))

    @pa.check("zip", error="Must start with numeric")
    def check_zip(cls, series: Series[str]) -> Series[bool]:
        return series.str.match(r"^\d")

    @pa.check("constraction_year", error="Year must be between 1800 and next year")
    def check_contraction_year(cls, series: Series[int]) -> Series[bool]:
        """Check if construction year is within reasonable range."""
        current_year = datetime.datetime.now().year
        return (series >= 1000) & (series <= current_year + 1)

    @pa.check("rent", error="Rent must match property features")
    def check_rent_plausibility(cls, series: Series[int]) -> Series[bool]:
        return series > 0
