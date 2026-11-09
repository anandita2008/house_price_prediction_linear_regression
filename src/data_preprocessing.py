def load_and_clean_data(file_path):
    # Load the CSV
    # Handle missing values
    # Remove duplicates
    # Convert size into numerical BHK
    # Convert total_sqft into numerical values

    return df

import pandas as pd
import re


# Convert total_sqft values into numerical square feet
import pandas as pd
import re

def convert_sqft(value):
    value = str(value).strip()

    # Handle ranges such as 2100 - 2850
    range_match = re.fullmatch(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", value)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2

    # Extract number and optional unit
    match = re.fullmatch(r"(\d+\.?\d*)\s*(.*?)", value)
    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2).lower().replace(".", "").strip()
    


    # Convert supported units into square feet
    conversions = {
        "": 1,
        "sq meter": 10.7639,
        "sq meters": 10.7639,
        "sq yards": 9,
        "acres": 43560,
        "cents": 435.6,
        "guntha": 1089,
        "grounds": 2400,
        "perch": 272.25
    }

    if unit not in conversions:
        return None

    return number * conversions[unit]


# Main preprocessing function
# Main preprocessing function
def load_and_clean_data(file_path="data/bengaluru_house_prices.csv"):
    # 1. Load dataset
    df = pd.read_csv(file_path)

    print("Original dataset shape:", df.shape)

    # 2. Remove duplicate rows
    df = df.drop_duplicates()

    # 3. Handle missing values in location
    df = df.dropna(subset=["location"])

    # 4. Convert size into numerical BHK
    df["bhk"] = pd.to_numeric(
    df["size"].astype(str).str.extract(r"(\d+)", expand=False), 
    errors="coerce"
    )

    # 5. Convert total_sqft into numerical values
    df["total_sqft"] = df["total_sqft"].apply(
        convert_sqft
    )

    df["total_sqft"] = pd.to_numeric(
        df["total_sqft"], errors="coerce"
    )

    # 6. Remove rows with essential missing values
    df = df.dropna(
        subset=["area_type","bhk", "total_sqft", "price"]
    )

    # 7. Fill missing numerical values
    df["bath"] = df["bath"].fillna(
        df["bath"].median()
    )

    df["balcony"] = df["balcony"].fillna(
        df["balcony"].median()
    )

    # 8. Remove society and original size columns
    # Society contains many missing values
    df = df.drop(
        columns=["society", "size"],
        errors="ignore"
    )

    # 9. Remove invalid values
    df = df[
        (df["total_sqft"] > 0) &
        (df["price"] > 0) &
        (df["bhk"] > 0)
    ]

    # 10. Reset row indexes
    df = df.reset_index(drop=True)

    print("\nData preprocessing completed!")
    print("Cleaned dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())

    return df
