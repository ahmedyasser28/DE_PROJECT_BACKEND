from pydantic import BaseModel
from typing import Optional, Any


class TrainRequest(BaseModel):
    task_type: str
    target_column: Optional[str] = None
    # n_clusters removed — now auto-detected


class MetricsResponse(BaseModel):
    task_type: str
    best_model: str
    metrics: dict[str, Any]
    all_models_metrics: dict[str, Any]
    optimal_k: Optional[int] = None          # clustering only
    k_scores: Optional[dict[str, Any]] = None  # silhouette per k


class UploadResponse(BaseModel):
    message: str
    filename: str
    rows: int
    columns: list[str]
    preview: list[dict]
