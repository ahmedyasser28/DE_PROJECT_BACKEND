# AutoML API

FastAPI project for uploading a dataset, training a simple machine learning pipeline, and exporting the trained artifact.

## Project Structure

```text
app/
├── __init__.py
├── main.py
├── state.py
├── routers/
│   ├── __init__.py
│   ├── upload.py
│   ├── train.py
│   └── export.py
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py
│   └── trainer.py
└── models/
    ├── __init__.py
    └── schemas.py
```

## Features

- Upload a `.csv` or `.xlsx` dataset.
- Store the uploaded dataset in in-memory app state.
- Train a model for `classification`, `regression`, or `clustering`.
- Export the trained model and preprocessing pipeline as a `.joblib` file.

## Requirements

- Python 3.14

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## API Endpoints

- `GET /health`
  Returns a simple health response.
- `POST /upload`
  Uploads a dataset file as multipart form-data using the `file` field.
- `POST /train`
  Trains a model using the uploaded dataset.
- `GET /export-model`
  Downloads the trained pipeline artifact as a `.joblib` file.

## Train Request Body

### Classification

```json
{
  "task_type": "classification",
  "target_column": "target"
}
```

### Regression

```json
{
  "task_type": "regression",
  "target_column": "price"
}
```

### Clustering

```json
{
  "task_type": "clustering",
  "n_clusters": 3
}
```

Notes:

- `task_type` must be `classification`, `regression`, or `clustering`.
- `target_column` is required for `classification` and `regression`.
- `target_column` is not used for `clustering`.
- `n_clusters` is only used for `clustering`.

## Docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Dependencies

The project currently uses:

- `fastapi`
- `uvicorn[standard]`
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `joblib`
- `openpyxl`
- `python-multipart`
