
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create outputs folder
os.makedirs("outputs", exist_ok=True)


def generate_data_plots(df, y_test, y_pred):

    # 1. House Price Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"].dropna(), bins=40, kde=True)
    plt.title("Distribution of House Prices")
    plt.xlabel("Price (Lakhs)")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.savefig("outputs/price_distribution.png")
    plt.show()
    plt.close()

    # 2. Total Area vs House Price
    if "total_sqft" in df.columns:
        area = pd.to_numeric(
            df["total_sqft"].astype(str).str.replace(",", ""),
            errors="coerce"
        )

        valid = area.notna() & df["price"].notna()

        plt.figure(figsize=(8, 5))
        plt.scatter(area[valid], df.loc[valid, "price"], alpha=0.4)
        plt.title("Total Area vs House Price")
        plt.xlabel("Total Area (Sq. Ft.)")
        plt.ylabel("Price (Lakhs)")
        plt.tight_layout()
        plt.savefig("outputs/area_vs_price.png")
        plt.show()
        plt.close()

    # 3. BHK vs House Price
       if "bhk" in df.columns:
           bhk = pd.to_numeric(df["bhk"], errors="coerce")

        valid = bhk.notna() & df["price"].notna()

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=bhk[valid], y=df.loc[valid, "price"])
        plt.title("BHK vs House Price")
        plt.xlabel("Number of Bedrooms (BHK)")
        plt.ylabel("Price (Lakhs)")
        plt.tight_layout()
        plt.savefig("outputs/bhk_vs_price.png")
        plt.show()
        plt.close()

    # 4. Correlation Heatmap
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        plt.figure(figsize=(9, 6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("outputs/correlation_heatmap.png")
        plt.show()
        plt.close()

    # 5. Actual vs Predicted House Prices
    actual = np.asarray(y_test).ravel()
    predicted = np.asarray(y_pred).ravel()

    plt.figure(figsize=(8, 5))
    plt.scatter(actual, predicted, alpha=0.5)

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Perfect Prediction"
    )

    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual Price (Lakhs)")
    plt.ylabel("Predicted Price (Lakhs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.show()
    plt.close()

    # 6. Residual Plot
    residuals = actual - predicted

    plt.figure(figsize=(8, 5))
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Price (Lakhs)")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig("outputs/residual_plot.png")
    plt.show()
    plt.close()

    print("All data plots generated successfully!")

