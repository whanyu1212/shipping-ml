#!/usr/bin/env python3
"""Example: Loading and validating CSV data with CSVLoader.

This example demonstrates how to use the CSVLoader to load rental data
from a CSV file with automatic schema validation.

Usage:
    python examples/load_csv_data.py
"""

from rental_prediction.data import CSVLoader, DataLoadError


def main():
    """Load and display rental data from CSV."""
    # Create a CSV loader with validation and caching enabled
    loader = CSVLoader("data/rent_apartments.csv", validate=True, cache=True)

    try:
        # Load the data
        data = loader.load()

        print("✅ Data loaded successfully!")
        print(f"\nDataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\nFirst few rows:")
        print(data.head())

        # Show some statistics
        print("\n" + "="*60)
        print("Summary Statistics:")
        print("="*60)
        print(f"Average rent: ${data['rent'].mean():.2f}")
        print(f"Average area: {data['area'].mean():.2f} m²")
        print(f"Oldest building: {data['construction_year'].min()}")
        print(f"Newest building: {data['construction_year'].max()}")

    except DataLoadError as e:
        print(f"❌ Error loading data: {e}")


if __name__ == "__main__":
    main()
