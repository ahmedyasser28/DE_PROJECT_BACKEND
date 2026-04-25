import joblib
import pandas as pd

artifact = joblib.load(r"C:\Users\mh616\Downloads\clustering_KMeans_pipeline.joblib")

print("Best model:", artifact["best_model_name"])

new_employees = pd.DataFrame({
    "Age":              [35, 50, 28, 45, 60],
    "Salary":           [85000, 45000, 95000, 60000, 110000],
    "Experience_Years": [10, 3, 15, 5, 35],
    "Department":       ["Engineering", "HR", "Engineering", "Sales", "Marketing"],
    "Training_Hours":   [80, 20, 90, 40, 70],
    "Performance_Score":[75.0, 45.0, 88.0, 52.0, 91.0],
    "Promoted":         [1, 0, 1, 0, 1],
})

X_new = artifact["preprocessor"].transform(new_employees)
clusters = artifact["model"].predict(X_new)

new_employees["Cluster"] = clusters
print("\n", new_employees[["Age", "Salary", "Experience_Years", "Cluster"]])