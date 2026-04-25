from fastapi import APIRouter, HTTPException
from app.models.schemas import TrainRequest, MetricsResponse
from app.pipeline.preprocessor import preprocess
from app.pipeline.trainer import train_classification, train_regression, train_clustering
from app.state import app_state

router = APIRouter(prefix="/train", tags=["Train"])


@router.post("", response_model=MetricsResponse)
def train_model(request: TrainRequest):
    """
    Run the full AutoML pipeline:
      1. Preprocess the uploaded dataset
      2. Train two models for the selected task
      3. Pick the best one and return evaluation metrics
    """

    # ── Guard: dataset must be uploaded first ───────────────────────────
    df = app_state.get("dataframe")
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset found. Please upload a file first via POST /upload.",
        )

    task = request.task_type.lower()
    if task not in ("classification", "regression", "clustering"):
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification', 'regression', or 'clustering'.",
        )

    if task in ("classification", "regression") and not request.target_column:
        raise HTTPException(
            status_code=400,
            detail="target_column is required for classification and regression tasks.",
        )

    # ── Guard: warn if continuous variable used for classification ────────
    if task == "classification" and request.target_column:
        target_col = df[request.target_column]
        n_unique = target_col.nunique()
        n_rows = len(df)
        if n_unique > 20 or (n_unique / n_rows) > 0.5:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The selected target column '{request.target_column}' appears to be a "
                    f"continuous variable ({n_unique} unique values). "
                    f"Classification requires categorical targets (e.g. Male/Female, Yes/No). "
                    f"Please use Regression instead."
                ),
            )

    # ── Guard: warn if categorical variable used for regression ──────────
    if task == "regression" and request.target_column:
        n_unique = df[request.target_column].nunique()
        is_numeric = df[request.target_column].dtype in ["int64", "float64", "int32", "float32"]
        if not is_numeric or n_unique <= 10:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The selected target column '{request.target_column}' appears to be a "
                    f"categorical variable ({n_unique} unique values). "
                    f"Regression requires a continuous numeric target (e.g. Price, Age, Score). "
                    f"Please use Classification instead."
                ),
            )

    # ── Preprocessing ────────────────────────────────────────────────────
    try:
        X, y, preprocessor, label_encoder, feature_names = preprocess(
            df=df,
            task_type=task,
            target_column=request.target_column,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    # ── Training & Evaluation ────────────────────────────────────────────
    try:
        if task == "classification":
            best_model, best_name, all_metrics = train_classification(X, y)
            best_metrics = all_metrics[best_name]

        elif task == "regression":
            best_model, best_name, all_metrics = train_regression(X, y)
            best_metrics = all_metrics[best_name]

        else:  # clustering
            n_clusters = request.n_clusters or 3
            best_model, best_name, all_metrics, labels = train_clustering(X, n_clusters)
            best_metrics = all_metrics[best_name]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    # ── Persist artifact for export ──────────────────────────────────────
    app_state["trained_artifact"] = {
        "model":          best_model,
        "preprocessor":   preprocessor,
        "label_encoder":  label_encoder,
        "feature_names":  feature_names,
        "task_type":      task,
        "best_model_name": best_name,
    }

    return MetricsResponse(
        task_type=task,
        best_model=best_name,
        metrics=best_metrics,
        all_models_metrics=all_metrics,
    )