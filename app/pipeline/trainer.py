
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score,
)


# ======================================================================== #
#  CLASSIFICATION
# ======================================================================== #

def train_classification(X, y):
    """
    Train Logistic Regression vs KNN Classifier.
    Winner = higher macro F1-Score on test set.
    Returns: best_model, best_name, all_metrics dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN Classifier":      KNeighborsClassifier(n_neighbors=5),
    }

    all_metrics = {}
    best_name, best_model, best_f1 = None, None, -1

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        avg = "binary" if len(np.unique(y)) == 2 else "macro"
        metrics = {
            "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":        round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "recall":           round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1_score":         round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        all_metrics[name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1   = metrics["f1_score"]
            best_name = name
            best_model = model

    return best_model, best_name, all_metrics


# ======================================================================== #
#  REGRESSION
# ======================================================================== #

def train_regression(X, y):
    """
    Train Linear Regression vs KNN Regressor.
    Winner = higher R² Score on test set.
    Returns: best_model, best_name, all_metrics dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor":     KNeighborsRegressor(n_neighbors=5),
    }

    all_metrics = {}
    best_name, best_model, best_r2 = None, None, -np.inf

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "MAE":  round(float(mean_absolute_error(y_test, y_pred)), 4),
            "MSE":  round(float(mean_squared_error(y_test, y_pred)), 4),
            "R2":   round(float(r2_score(y_test, y_pred)), 4),
        }
        all_metrics[name] = metrics

        if metrics["R2"] > best_r2:
            best_r2   = metrics["R2"]
            best_name = name
            best_model = model

    return best_model, best_name, all_metrics


# ======================================================================== #
#  CLUSTERING
# ======================================================================== #

def train_clustering(X, n_clusters: int = 3):
    """
    Train KMeans vs Agglomerative Clustering.
    Winner = higher Silhouette Score.
    Returns: best_model, best_name, all_metrics dict, best_labels
    """
    candidates = {
        "KMeans":                  KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
    }

    all_metrics = {}
    best_name, best_model, best_score, best_labels = None, None, -1, None

    for name, model in candidates.items():
        labels = model.fit_predict(X)
        # Silhouette requires at least 2 distinct labels
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            sil = -1.0
        else:
            sil = round(float(silhouette_score(X, labels)), 4)

        all_metrics[name] = {"silhouette_score": sil}

        if sil > best_score:
            best_score  = sil
            best_name   = name
            best_model  = model
            best_labels = labels

    return best_model, best_name, all_metrics, best_labels