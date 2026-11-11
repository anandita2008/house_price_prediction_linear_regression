
# Import required libraries
from plots import generate_data_plots

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def train_and_evaluate(df):

    # Convert BHK into numeric values
    df["bhk"] = pd.to_numeric(
        df["bhk"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

    # Convert total_sqft into numeric values
    df["total_sqft"] = pd.to_numeric(
        df["total_sqft"],
        errors="coerce"
    )

    # Select features
    X = df[
        ["area_type", "location", "bhk",
         "total_sqft", "bath", "balcony"]
    ]

    # Target variable
    y = df["price"]

    # Convert categorical data into numerical data
    X = pd.get_dummies(
        X,
        columns=["area_type", "location"],
        drop_first=True,
        dtype=int
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle missing values
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Save model
    with open("outputs/linear_regression_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Save predictions
    predictions = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": y_pred
    })

    predictions.to_csv("outputs/predictions.csv", index=False)

    # Save evaluation results
    with open("outputs/results.txt", "w") as file:
        file.write("Linear Regression Model Results\n")
        file.write("--------------------------------\n")
        file.write(f"MAE: {mae:.4f}\n")
        file.write(f"MSE: {mse:.4f}\n")
        file.write(f"RMSE: {rmse:.4f}\n")
        file.write(f"R2 Score: {r2:.4f}\n")

    # Generate graphs
    generate_data_plots(df, y_test, y_pred)

    # Print results
    print("\nModel Evaluation Results")
    print("------------------------")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print("\nModel saved successfully!")
    print("Predictions saved successfully!")
    print("Results saved successfully!")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }

