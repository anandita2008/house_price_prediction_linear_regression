
import pandas as pd

from model_training import train_and_evaluate

# Load dataset
df = pd.read_csv("data/bengaluru_house_prices.csv")

# Train and evaluate Linear Regression model
results = train_and_evaluate(df)

print("\nAll graphs generated successfully!")

