
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse
from app.state import app_state

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or XLSX dataset.
    The dataframe is stored in memory (app_state) for the training step.
    """
    filename = file.filename or ""

    # ── Validate extension ──────────────────────────────────────────────
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .csv or .xlsx file.",
        )

    contents = await file.read()

    # ── Parse file ──────────────────────────────────────────────────────
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    # ── Store in shared state ───────────────────────────────────────────
    app_state["dataframe"] = df
    app_state["trained_artifact"] = None   # reset any previous training

    # ── Build preview (first 5 rows, NaN → None for JSON safety) ────────
    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")

    return UploadResponse(
        message="File uploaded successfully.",
        filename=filename,
        rows=len(df),
        columns=df.columns.tolist(),
        preview=preview,
    )