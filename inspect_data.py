
import pandas as pd

# Load the original survey
url = (
    "https://raw.githubusercontent.com/rfordatascience/"
    "tidytuesday/main/data/2024/2024-05-14/"
    "coffee_survey.csv"
)

df = pd.read_csv(url)

# Our target and candidate features
columns = ["roast_level", "brew", "strength", "style", "cups"]

print("DATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df[columns].isna().sum())

print("\nMISSING PERCENTAGES")
print((df[columns].isna().mean() * 100).round(1))

print("\nACTUAL VALUES AND COUNTS")
for col in columns:
    print(f"\n{col.upper()}")
    print(df[col].value_counts(dropna=False).to_string())

print("\nCOMPLETE ROWS")
complete = df[columns].dropna()
print(f"{len(complete)} of {len(df)}")

print("\nROAST LEVEL CLASS DISTRIBUTION")
print(df["roast_level"].value_counts(normalize=True).mul(100).round(1))
