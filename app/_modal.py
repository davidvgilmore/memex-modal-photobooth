import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.routing import Mount
from modal import App, Image, Secret, Volume, asgi_app, gpu
from pydantic import UUID4

from app.photobooth.schemas import LoraCreate, PhotoboothCreate
from app.settings import settings

app = App(name="modal-photobooth")
volume = Volume.from_name("modal-photobooth-data", create_if_missing=True)

_app_env_dict: Dict[str, str | None] = {
    f"APP_{str(k)}": str(v) for k, v in json.loads(settings.model_dump_json()).items()
}
_remove_prefix_keys = [
    "APP_HF_TOKEN",
    "APP_HF_HUB_ENABLE_HF_TRANSFER",
]
for key in _remove_prefix_keys:
    if key in _app_env_dict:
        new_key = key.replace("APP_", "")
        _app_env_dict[new_key] = _app_env_dict[key]
app_env = Secret.from_dict(_app_env_dict)

DIFFUSERS_GIT_SHA = "e8aacda762e311505ba05ae340af23b149e37af3"
VOL_DIR = "/root/modal-photobooth-data"

image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(["uv"])
    .workdir("/root")
    .copy_local_file("pyproject.toml", "/root/pyproject.toml")
    .copy_local_file("uv.lock", "/root/uv.lock")
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .run_commands(
        [
            "uv sync --no-dev --frozen --compile-bytecode",
            "uv build",
        ]
    )
    .workdir("/root/diffusers")
    .run_commands(
        "git init .",
        "git remote add origin https://github.com/huggingface/diffusers",
        f"git fetch --depth=1 origin {DIFFUSERS_GIT_SHA}",
        f"git checkout {DIFFUSERS_GIT_SHA}",
        "pip install -e .",
    )
)

parent = Path(__file__).parent.parent
templates_path = parent / "templates"


@app.function(
    container_idle_timeout=30,
    image=image,
    secrets=[app_env],
    timeout=600,  # 10 minutes
    volumes={VOL_DIR: volume},
    mounts=[
        Mount("/root/templates", local_dir=templates_path),
    ],
)
@asgi_app(label="photobooth-server")
def api_server() -> FastAPI:
    from app.main import app

    return app


@app.function(
    image=image,
    gpu=gpu.H100(count=1),
    volumes={VOL_DIR: volume},
    timeout=2400,  # 40 minutes
    secrets=[app_env],
)
def trainer(lora_id: UUID4, lora_create: LoraCreate) -> None:
    from app.photobooth.service import PhotoboothService

    volume.reload()

    PhotoboothService()._train_lora(lora_id, lora_create)


@app.function(
    image=image,
    gpu=gpu.H100(count=1),
    volumes={VOL_DIR: volume},
    timeout=1800,
    secrets=[app_env],
)
def inference(photobooth_create: PhotoboothCreate) -> None:
    from app.photobooth.service import PhotoboothService

    volume.reload()

    PhotoboothService()._run_inference(photobooth_create=photobooth_create)
