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
    class_counts = np.bincount(y)
    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
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
            best_f1    = metrics["f1_score"]
            best_name  = name
            best_model = model

    return best_model, best_name, all_metrics


# ======================================================================== #
#  REGRESSION
# ======================================================================== #

def train_regression(X, y):
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
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "MSE": round(float(mean_squared_error(y_test, y_pred)), 4),
            "R2":  round(float(r2_score(y_test, y_pred)), 4),
        }
        all_metrics[name] = metrics

        if metrics["R2"] > best_r2:
            best_r2    = metrics["R2"]
            best_name  = name
            best_model = model

    return best_model, best_name, all_metrics


# ======================================================================== #
#  CLUSTERING  —  each algorithm finds its own optimal k via silhouette score
# ======================================================================== #

def find_optimal_k_kmeans(X, k_min: int = 2, k_max: int = 10):
    """Find optimal k for KMeans using silhouette scores."""
    k_max = min(k_max, len(X) - 1)
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
        else:
            scores[k] = round(float(silhouette_score(X, labels)), 4)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def find_optimal_k_agglomerative(X, k_min: int = 2, k_max: int = 10):
    """Find optimal k for Agglomerative Clustering using silhouette scores."""
    k_max = min(k_max, len(X) - 1)
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
        else:
            scores[k] = round(float(silhouette_score(X, labels)), 4)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def train_clustering(X):
    """
    Each algorithm independently finds its own optimal k using silhouette scores (k=2..10).
    Winner = higher Silhouette Score at their respective optimal k.
    Returns: best_model, best_name, all_metrics dict, best_labels, optimal_k, k_scores
    """
    # Step 1 — find optimal k separately for each algorithm
    kmeans_k, kmeans_k_scores = find_optimal_k_kmeans(X)
    agglo_k,  agglo_k_scores  = find_optimal_k_agglomerative(X)

    # Step 2 — train each algorithm with its own optimal k
    candidates = {
        "KMeans": (
            KMeans(n_clusters=kmeans_k, random_state=42, n_init=10),
            kmeans_k,
            kmeans_k_scores,
        ),
        "Agglomerative Clustering": (
            AgglomerativeClustering(n_clusters=agglo_k),
            agglo_k,
            agglo_k_scores,
        ),
    }

    all_metrics = {}
    best_name, best_model, best_score, best_labels = None, None, -1, None
    best_k, best_k_scores = None, None

    for name, (model, optimal_k, k_scores) in candidates.items():
        labels = model.fit_predict(X)
        n_unique = len(np.unique(labels))
        sil = round(float(silhouette_score(X, labels)), 4) if n_unique >= 2 else -1.0

        all_metrics[name] = {
            "silhouette_score": sil,
            "optimal_k":        optimal_k,
        }

        if sil > best_score:
            best_score    = sil
            best_name     = name
            best_model    = model
            best_labels   = labels
            best_k        = optimal_k
            best_k_scores = k_scores

    return best_model, best_name, all_metrics, best_labels, best_k, best_k_scores
