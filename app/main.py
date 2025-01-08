import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict

import structlog
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.templating import Jinja2Templates

from app import __version__
from app.kit.sqlite import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    create_async_sessionmaker,
)
from app.logging import configure_logging
from app.middlewares import setup_middlewares
from app.photobooth.schemas import ModelName
from app.photobooth.service import PhotoboothService
from app.router import router

log = structlog.get_logger()

os.environ["TZ"] = "UTC"


def generate_unique_openapi_id(route: APIRoute) -> str:
    return f"{route.tags[0]}:{route.name}"


class State(TypedDict):
    asyncengine: AsyncEngine
    asyncsessionmaker: async_sessionmaker[AsyncSession]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    log.debug("starting app lifespan...")
    log.debug("creating async engine")
    asyncengine = create_async_engine("app")
    log.debug("created async sessionmaker")
    log.debug("creating async sessionmaker")
    asyncsessionmaker = create_async_sessionmaker(asyncengine)
    log.debug("created async sessionmaker")
    log.debug("downloading models")
    pb_svc = PhotoboothService()
    for model_name in ModelName:
        log.debug(f"downloading {model_name}...")
        try:
            pb_svc._download_models(model_name)
            log.debug(f"downloaded {model_name}")
        except Exception as e:
            log.error(f"failed to download {model_name}", error=str(e))
    log.debug("downloaded models")
    log.debug("serving state")
    yield {"asyncengine": asyncengine, "asyncsessionmaker": asyncsessionmaker}
    log.debug("disposing async sessionmaker")
    await asyncengine.dispose()
    log.debug("disposed async engine")
    log.debug("... app lifespan ended")


def create_app() -> FastAPI:
    app = FastAPI(
        title="modal-photobooth",
        generate_unique_id_function=generate_unique_openapi_id,
        version=__version__,
        lifespan=lifespan,
    )

    setup_middlewares(app)
    app.include_router(router)

    return app


configure_logging()

templates = Jinja2Templates(directory="templates")
app = create_app()
