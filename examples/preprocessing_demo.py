#!/usr/bin/env python3
"""Example: Preprocessing pipeline demonstration.

This example demonstrates how the DataPreprocessor transforms raw rental data through
a chain of transformations:
1. Pruning unnecessary columns (address, zip, etc.)
2. Encoding categorical columns (balcony, parking, etc.)
3. Parsing garden area from text to numeric

Usage:
    python examples/preprocessing_demo.py
"""

from rental_prediction.data import CSVLoader
from rental_prediction.preprocessor.data_preprocessor import DataPreprocessor


def main():
    """Load data, apply preprocessing, and show before/after comparison."""
    # Load the data
    loader = CSVLoader("data/rent_apartments.csv", validate=True)
    data = loader.load()

    print("=" * 80)
    print("BEFORE PREPROCESSING")
    print("=" * 80)
    print(f"\nDataset shape: {data.shape}")
    print(f"\nColumns ({len(data.columns)}): {list(data.columns)}")
    print("\nCategorical columns (sample values):")
    categorical_cols = ["balcony", "parking", "furnished", "garage", "storage"]
    for col in categorical_cols:
        if col in data.columns:
            print(f"  {col}: {data[col].unique()[:3]}")

    print("\nGarden column (sample values):")
    print(f"  {data['garden'].unique()[:5]}")

    print("\nFirst 3 rows (selected columns):")
    display_cols = [
        "area",
        "rooms",
        "balcony",
        "parking",
        "furnished",
        "garden",
        "rent",
    ]
    print(data[display_cols].head(3))

    # Apply preprocessing
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process(data)

    print("\n" + "=" * 80)
    print("AFTER PREPROCESSING")
    print("=" * 80)
    print(f"\nDataset shape: {processed_data.shape}")
    print(f"\nColumns ({len(processed_data.columns)}): {list(processed_data.columns)}")

    print("\nEncoded categorical columns:")
    encoded_cols = [
        col
        for col in processed_data.columns
        if any(cat in col for cat in categorical_cols)
    ]
    print(f"  {encoded_cols}")

    print("\nGarden column (now numeric):")
    print(f"  Type: {processed_data['garden'].dtype}")
    print(f"  Sample values: {processed_data['garden'].unique()[:5]}")

    print("\nFirst 3 rows (selected columns):")
    display_cols_after = ["area", "rooms", "garden", "rent"]
    # Add some encoded columns for demonstration
    encoded_display = [col for col in encoded_cols[:4] if col in processed_data.columns]
    print(processed_data[display_cols_after + encoded_display].head(3))

    print("\n" + "=" * 80)
    print("TRANSFORMATION SUMMARY")
    print("=" * 80)

    # Show pruned columns
    pruned_columns = set(data.columns) - set(processed_data.columns)
    # Exclude encoded columns from pruned count
    actually_pruned = [col for col in pruned_columns if col not in categorical_cols]
    if actually_pruned:
        print(f"✅ Pruned {len(actually_pruned)} unnecessary columns: {actually_pruned}")

    print(f"✅ One-hot encoded {len(categorical_cols)} categorical columns")
    print(f"✅ Parsed garden area from text to numeric (m²)")
    print(
        f"✅ Net change: {data.shape[1]} → {processed_data.shape[1]} columns"
    )

    # Check for missing values
    missing_before = data.isnull().sum().sum()
    missing_after = processed_data.isnull().sum().sum()
    if missing_after > 0:
        print(f"⚠️  Missing values detected: {missing_after} total")
        cols_with_missing = processed_data.isnull().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0]
        for col, count in cols_with_missing.items():
            print(f"   - {col}: {count} missing")
    else:
        print(
            f"✅ No missing values (before: {missing_before}, after: {missing_after})"
        )

    print(f"✅ Data ready for model training!")


if __name__ == "__main__":
    main()
