
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

    df["bhk"] = pd.to_numeric(
<<<<<<< Updated upstream
        df["bhk"],
=======
        df["bhk"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
>>>>>>> Stashed changes
        errors="coerce"
    )

    df["total_sqft"] = pd.to_numeric(
        df["bhk"],
        errors="coerce"
    )

    X = df[
        ["area_type", "location", "bhk",
         "total_sqft", "bath", "balcony"]
    ]

    y = df["price"]

    # 2. Convert categorical data into numerical data
    X = pd.get_dummies(
        X,
        columns=["area_type", "location"],
        drop_first=True,
        dtype=int
    )

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 4. Handle missing values
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # 5. Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model training completed successfully!")

    # 6. Predict house prices
    y_pred = model.predict(X_test)

    # 7. Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 8. Display results
    print("\nMODEL EVALUATION")
    print("-------------------------")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R2 Score:", r2)

    # 9. Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # 10. Save evaluation metrics
    with open("outputs/results.txt", "w") as file:
        file.write("MODEL EVALUATION RESULTS\n")
        file.write("-------------------------\n")
        file.write(f"MAE: {mae}\n")
        file.write(f"MSE: {mse}\n")
        file.write(f"RMSE: {rmse}\n")
        file.write(f"R2 Score: {r2}\n")

    # 11. Save actual and predicted prices
    predictions = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": y_pred
    })

    predictions.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    # 12. Save trained model
    with open("outputs/linear_regression_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # 13. Generate graphs
    generate_data_plots(df, y_test, y_pred)

    # 14. Return results
    results = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    print("\nAll outputs saved successfully in outputs folder!")

    return results
