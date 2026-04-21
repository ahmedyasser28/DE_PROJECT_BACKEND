
import io
import joblib
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.state import app_state
router = APIRouter(prefix="/export-model", tags=["Export"])


@router.get("")
def export_model():
    """
    Serialize and download the trained model + preprocessing pipeline
    as a single .joblib file.

    The artifact contains:
        {
            "model":           <trained sklearn model>,
            "preprocessor":    <fitted ColumnTransformer>,
            "label_encoder":   <fitted LabelEncoder or None>,
            "feature_names":   <list of feature names>,
            "task_type":       <str>,
            "best_model_name": <str>,
        }
    """
    artifact = app_state.get("trained_artifact")

    if artifact is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Please run POST /train first.",
        )

    # ── Serialize to in-memory bytes ─────────────────────────────────────
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    buffer.seek(0)

    model_name = artifact["best_model_name"].replace(" ", "_")
    filename   = f"{artifact['task_type']}_{model_name}_pipeline.joblib"

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )