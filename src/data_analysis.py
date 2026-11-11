import pandas as pd


def analyze_data(df):

    # 1. Dataset 
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
#I have verified data analysis part.
#I also tried running all codes on my windows pc and they are running perfectly
#earlier it was having issues in my laptop but now everything is running properly
