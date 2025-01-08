import asyncio
import base64
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import structlog
from fastapi import APIRouter, Body, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from modal import Function
from pydantic import UUID4

from app._modal import volume
from app.kit.sqlite import AsyncSession, get_async_db_session
from app.photobooth.schemas import LoraCreate, LoraCreateResponse, PhotoboothCreate
from app.photobooth.service import PhotoboothService

router = APIRouter(tags=["photobooth"])
log = structlog.get_logger()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


class Routes:
    home = "/"
    create_lora = "/lora"
    retrain_lora = "/lora/{lora_id}/retrain"
    create_photobooth = "/photobooth"


@router.get(Routes.home, response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    images: Dict[str, List[str]] = defaultdict(list)

    async def process_results_entry(
        results_entry: Any, lora_id: str, images: Dict[str, List[str]]
    ) -> None:
        if (
            results_entry.type.value == 1
            and os.path.splitext(results_entry.path)[1] != ".safetensors"
        ):
            image_data = BytesIO()
            volume.read_file_into_fileobj(results_entry.path, image_data)
            bytes_value = image_data.getvalue()
            images[lora_id].append(
                f'data:image/png;base64,{base64.b64encode(bytes_value).decode("utf-8")}'
            )

    async def process_dir_entry(dir_entry: Any, images: Dict[str, List[str]]) -> None:
        tasks = []
        for results_entry in volume.listdir(f"/{dir_entry.path}/results"):
            lora_id = results_entry.path.split("/")[1]
            tasks.append(process_results_entry(results_entry, lora_id, images))
        await asyncio.gather(*tasks)

    try:
        tasks = [
            process_dir_entry(dir_entry, images)
            for dir_entry in volume.listdir("/content")
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        log.error("failed to process images", error=str(e))

    return templates.TemplateResponse(
        name="photobooth_home.html", context={"request": request, "images": images}
    )


@router.post(Routes.create_lora, response_model=LoraCreateResponse)
async def create_lora(
    lora_create: LoraCreate = Body(...),
    content: UploadFile = File(...),
    pb_svc: PhotoboothService = Depends(PhotoboothService),
    session: AsyncSession = Depends(get_async_db_session),
) -> LoraCreateResponse:
    log.debug("received request to create lora", lora_create=lora_create)
    log.debug("received content to create lora", content=content)
    lora_id = uuid4()
    pb_svc._write_training_images(
        lora_id=lora_id,
        content=content,
        resolution=lora_create.tuning_config.hf_hyperparameters.resolution,
    )
    Function.lookup("modal-photobooth", "trainer").spawn(
        lora_id=lora_id, lora_create=lora_create
    )
    return LoraCreateResponse(detail="training your custom lora!", lora_id=lora_id)


@router.post(Routes.retrain_lora, response_model=LoraCreateResponse)
async def retrain_lora(
    lora_id: UUID4,
    lora_create: LoraCreate = Body(...),
    session: AsyncSession = Depends(get_async_db_session),
) -> LoraCreateResponse:
    log.debug("received request to retrain lora", lora_id=lora_id)
    Function.lookup("modal-photobooth", "trainer").spawn(
        lora_id=lora_id, lora_create=lora_create
    )
    return LoraCreateResponse(detail="retraining your custom lora!", lora_id=lora_id)


@router.post(Routes.create_photobooth, response_model=None)
async def create_photobooth(
    photobooth_create: PhotoboothCreate = Body(...),
    pb_svc: PhotoboothService = Depends(PhotoboothService),
    session: AsyncSession = Depends(get_async_db_session),
) -> None:
    log.debug(
        "received request to create photobooth", photobooth_create=photobooth_create
    )
    Function.lookup("modal-photobooth", "inference").spawn(
        photobooth_create=photobooth_create
    )
    return None
