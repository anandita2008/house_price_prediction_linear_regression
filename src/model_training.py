
# Import required libraries
from plots import generate_data_plots
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Main model training function

def train_and_evaluate(df):

       # 1. Select input features and target variable

    df["bhk"] = pd.to_numeric(
        df["size"].str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )
    df["total_sqft"] = pd.to_numeric(
    df["total_sqft"].astype(str).str.split("-").str[0],
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


    # 4. Create Linear Regression model

    model = LinearRegression()


    # 5. Train the model
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_train.median(numeric_only=True))
    model.fit(X_train, y_train)

    print("Model training completed successfully!")


    # 6. Predict house prices

    y_pred = model.predict(X_test)


    # 7. Evaluate model performance

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)


    # 8. Display evaluation results

    print("\nMODEL EVALUATION")
    print("-------------------------")

    print("Mean Absolute Error:", mae)

    print("Mean Squared Error:", mse)

    print("Root Mean Squared Error:", rmse)

    print("R2 Score:", r2)


    # 9. Return results

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
    # Generate plots
    generate_data_plots(df, y_test, y_pred)
    return results
