#this folder is for if we want to see the coefficients and all of our model
import joblib

model = joblib.load("outputs/linear_regression_model.pkl")

print("Model loaded successfully!")
print("\nModel type:", type(model).__name__)
print("\nCoefficients:", model.coef_)
print("\nIntercept:", model.intercept_)

if hasattr(model, "feature_names_in_"):
    print("\nFeatures:", model.feature_names_in_)
