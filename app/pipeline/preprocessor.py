
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def preprocess(
    df: pd.DataFrame,
    task_type: str,
    target_column: str = None,
) -> tuple:
    """
    Full preprocessing pipeline.

    Returns:
        - X_processed (np.ndarray)
        - y_processed (np.ndarray or None for clustering)
        - preprocessor (fitted ColumnTransformer — saved with model)
        - label_encoder (fitted LabelEncoder for classification target or None)
        - feature_names (list of output feature names)
    """

    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Split features / target
    # ------------------------------------------------------------------ #
    if task_type in ("classification", "regression"):
        if target_column is None or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        X = df.drop(columns=[target_column])
        y_raw = df[target_column].copy()
    else:
        # clustering — no target
        X = df.copy()
        y_raw = None

    # ------------------------------------------------------------------ #
    # 2. Drop columns that are entirely null
    # ------------------------------------------------------------------ #
    X = X.dropna(axis=1, how="all")

    # ------------------------------------------------------------------ #
    # 3. Identify numerical vs categorical columns
    # ------------------------------------------------------------------ #
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ------------------------------------------------------------------ #
    # 4. Build ColumnTransformer
    #    - Numerical  : median imputation → StandardScaler
    #    - Categorical: most-frequent imputation → OneHotEncoder
    # ------------------------------------------------------------------ #
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X)

    # ------------------------------------------------------------------ #
    # 5. Build feature names (useful for downstream reporting)
    # ------------------------------------------------------------------ #
    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]["encoder"]
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

    # ------------------------------------------------------------------ #
    # 6. Encode target
    # ------------------------------------------------------------------ #
    label_encoder = None
    y_processed = None

    if task_type == "classification" and y_raw is not None:
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y_raw.astype(str))

    elif task_type == "regression" and y_raw is not None:
        # Impute any missing target values with median
        target_imputer = SimpleImputer(strategy="median")
        y_processed = target_imputer.fit_transform(
            y_raw.values.reshape(-1, 1)
        ).ravel()

    # ------------------------------------------------------------------ #
    # 7. Handle class imbalance (classification only) using SMOTE
    # ------------------------------------------------------------------ #
    if task_type == "classification" and y_processed is not None:
        class_counts = np.bincount(y_processed)
        # Apply SMOTE only when minority class has at least 2 samples
        if len(class_counts) >= 2 and class_counts.min() >= 2:
            min_samples = class_counts.min()
            k_neighbors = min(5, min_samples - 1)
            try:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_processed, y_processed = smote.fit_resample(X_processed, y_processed)
            except Exception:
                # If SMOTE still fails just continue without it
                pass

    return X_processed, y_processed, preprocessor, label_encoder, feature_names