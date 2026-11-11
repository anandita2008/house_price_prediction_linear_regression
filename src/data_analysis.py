import pandas as pd


def analyze_data(df):

    # 1. Dataset overview
    print("Dataset Shape:", df.shape)

    print("\nDataset Information:")
    df.info()

    # 2. Statistical analysis
    print("\nStatistical Summary:")
    print(df.describe())

    # 3. Location analysis
    print("\nTop 10 Locations:")
    print(df["location"].value_counts().head(10))

    # 4. Correlation analysis
    numeric_df = df.select_dtypes(include="number")

    print("\nCorrelation with House Price:")
    correlation = numeric_df.corr()["price"].sort_values(
        ascending=False
    )

    print(correlation)

    return correlation
#I have completed data analysis part
