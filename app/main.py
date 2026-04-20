from fastapi import FastAPI

from app.routers.export import router as export_router
from app.routers.train import router as train_router
from app.routers.upload import router as upload_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoML API",
        version="0.1.0",
    )

    @app.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(upload_router)
    app.include_router(train_router)
    app.include_router(export_router)
    return app


app = create_app()
