import joblib
import pandas as pd

artifact = joblib.load(r"C:\Users\mh616\Downloads\model (7).joblib")

print("Best model:", artifact["best_model_name"])

new_employees = pd.DataFrame({
    "Age":              [35, 50, 28, 45, 60],
    "Salary":           [85000, 45000, 95000, 60000, 110000],
    "Experience_Years": [10, 3, 15, 5, 35],
    "Department":       ["Engineering", "HR", "Engineering", "Sales", "Marketing"],
    "Training_Hours":   [80, 20, 90, 40, 70],
    "Promoted":         [1, 0, 1, 0, 1],
})

X_new = artifact["preprocessor"].transform(new_employees)
predictions = artifact["model"].predict(X_new)

new_employees["Predicted_Performance_Score"] = predictions.round(2)
print("\n", new_employees[["Age", "Experience_Years", "Training_Hours", "Predicted_Performance_Score"]])