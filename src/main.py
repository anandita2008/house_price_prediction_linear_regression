import os
from data_preprocessing import load_and_clean_data
from data_analysis import analyze_data
from model_training import train_and_evaluate


def main():
    print("=" * 50)
    print("  BENGALURU HOUSE PRICE PREDICTION PIPELINE  ")
    print("=" * 50)

    # 1. Dataset path configuration
    data_path = "data/bengaluru_house_prices.csv"

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'. Check your file path!")
        return

    # 2. Data Preprocessing
    print("\n[STEP 1] Preprocessing dataset...")
    df = load_and_clean_data(data_path)

    # 3. Data Analysis
    print("\n[STEP 2] Running data analysis...")
    analyze_data(df)

    # 4. Model Training & Evaluation
    print("\n[STEP 3] Training linear regression model...")
    results = train_and_evaluate(df)

    print("\n" + "=" * 50)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()