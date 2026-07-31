"""Application factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Hybrid (keyword + vector) search and RAG over a document corpus.",
    )
    app.include_router(router)
    return app


app = create_app()
