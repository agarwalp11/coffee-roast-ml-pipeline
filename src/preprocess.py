
from pathlib import Path
import pandas as pd

# Define project paths
# path allows for reproductibility across different machines and operating systems
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "coffee_survey.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "coffee_cleaned.csv"

# Columns needed for our ML problem
FEATURES = ["brew", "strength", "style", "cups"]
TARGET = "roast_level"

# function to preprocess the data
def preprocess_data():
    # 1. Load original dataset
    df = pd.read_csv(RAW_PATH)
    print(f"Original dataset shape: {df.shape}")

    # 2. Keep only relevant columns
    df = df[FEATURES + [TARGET]].copy()

    # 3. Keep all survey responses
    # Identical answers may come from different respondents.
    # Do not remove duplicates based only on these five columns.

    # 4. Remove rows without a target
    df = df.dropna(subset=[TARGET])

    # category consolidation
    # standarize roast categories to "Light", "Medium", and "Dark"
    roast_mapping = {
        "Blonde": "Light",
        "Nordic": "Light",
        "Italian": "Dark",
        "French": "Dark"
    }

    df[TARGET] = df[TARGET].replace(roast_mapping)

    # 5. Handle missing predictor values
    df[FEATURES] = df[FEATURES].fillna("Unknown")

    # 6. Remove extra whitespace in text columns
    for column in FEATURES + [TARGET]:
        df[column] = df[column].astype(str).str.strip()

    # 7. Save cleaned dataset to processed folder
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nTarget distribution:")
    print(df[TARGET].value_counts())

    return df


if __name__ == "__main__":
    preprocess_data()
