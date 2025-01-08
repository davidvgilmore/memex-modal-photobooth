from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.settings import settings

log = structlog.get_logger()


def add_cors_middleware(app: FastAPI) -> None:
    log.debug("Adding CORS middleware", allow_origins=settings.CORS_ORIGINS or "None")
    allowed_origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else []
    return (
        None
        if not allowed_origins
        else app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    )


def add_session_middleware(app: FastAPI) -> None:
    app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)


def add_processing_time(app: FastAPI) -> None:
    @app.middleware("http")
    async def processing_time_ms(request: Request, call_next: Callable) -> Response:
        import time

        start_time = round(time.time() * 1000, 4)
        response = await call_next(request)
        process_time = round(round(time.time() * 1000, 4) - start_time, 4)
        response.headers["x-processing-time-ms"] = str(process_time)
        return response


def setup_middlewares(app: FastAPI) -> None:
    add_cors_middleware(app)
    add_session_middleware(app)
    add_processing_time(app)
