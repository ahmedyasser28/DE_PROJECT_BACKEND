import joblib
import pandas as pd

artifact = joblib.load(r"C:\Users\mh616\Downloads\model (6).joblib")

print("Best model:", artifact["best_model_name"])
print("Features:", artifact["feature_names"])

new_employees = pd.DataFrame({
    "Age":              [35, 50, 28, 45, 60],
    "Salary":           [85000, 45000, 95000, 60000, 110000],
    "Experience_Years": [10, 3, 15, 5, 35],
    "Department":       ["Engineering", "HR", "Engineering", "Sales", "Marketing"],
    "Training_Hours":   [80, 20, 90, 40, 70],
    "Performance_Score":[75.0, 45.0, 88.0, 52.0, 91.0],
})

X_new = artifact["preprocessor"].transform(new_employees)
predictions = artifact["model"].predict(X_new)
decoded = artifact["label_encoder"].inverse_transform(predictions)

new_employees["Predicted_Promoted"] = decoded
print("\n", new_employees[["Age", "Experience_Years", "Performance_Score", "Predicted_Promoted"]])