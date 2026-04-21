
from pydantic import BaseModel
from typing import Optional, Any


class TrainRequest(BaseModel):
    task_type: str          # "classification", "regression", "clustering"
    target_column: Optional[str] = None   # required for classification & regression
    n_clusters: Optional[int] = 3         # only for clustering


class MetricsResponse(BaseModel):
    task_type: str
    best_model: str
    metrics: dict[str, Any]
    all_models_metrics: dict[str, Any]


class UploadResponse(BaseModel):
    message: str
    filename: str
    rows: int
    columns: list[str]
    preview: list[dict]   # first 5 rows