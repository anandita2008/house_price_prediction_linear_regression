#main program for house price prediction using linear regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

#verify that the libraries are installed and working
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nAll four libraries are working!")



import pandas as pd

# Load the dataset
df = pd.read_csv("data/bengaluru_house_prices.csv")

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())


# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
df["location"] = df["location"].fillna(df["location"].mode()[0])
df["size"] = df["size"].fillna(df["size"].mode()[0])
df["society"] = df["society"].fillna("Unknown")
df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(df["balcony"].median())

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())
