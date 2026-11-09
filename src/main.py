
import pandas as pd

from model_training import train_and_evaluate

# Load dataset
df = pd.read_csv("data/Bengaluru_House_Data.csv")

# Train and evaluate Linear Regression model
results = train_and_evaluate(df)

print("\nAll graphs generated successfully!")

