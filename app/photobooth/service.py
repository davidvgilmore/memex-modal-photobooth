import os
import subprocess
from uuid import uuid4
from zipfile import ZipFile

import cv2
import structlog
import torch
from accelerate.utils import write_basic_config
from cv2 import data as cv2_data
from diffusers import DiffusionPipeline
from fastapi import UploadFile
from huggingface_hub import snapshot_download
from PIL import Image
from pydantic import UUID4

from app.photobooth.schemas import (
    LoraCreate,
    ModelName,
    PhotoboothCreate,
    SafetensorsFilename,
)
from app.settings import settings

log = structlog.get_logger()


class PhotoboothService:
    def _download_models(self, model_name: ModelName) -> None:
        # if the flux1 dev model is already downloaded, skip
        fname = getattr(SafetensorsFilename, model_name.name)
        if os.path.exists(f"{settings.MODEL_DIR}/{fname.value}"):
            log.debug(f"{model_name} already downloaded")
            return

        log.debug("downloading models...")
        snapshot_download(
            model_name.value,
            local_dir=settings.MODEL_DIR,
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        )
        log.debug("models downloaded")
        log.debug("downloading safetensors...")
        DiffusionPipeline.from_pretrained(
            settings.MODEL_DIR, torch_dtype=torch.bfloat16
        )
        log.debug("safetensors downloaded")

    def _exec_subprocess(self, cmd: list[str]) -> None:
        log.debug("PhotoboothService._exec_subprocess", cmd=cmd)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if process.stdout is None:
            raise ValueError("stdout is None")

        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                log.debug(line_str.strip())

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    def _format_image(
        self, image_path: str, resolution: int, padding_ratio: float = 0.4
    ) -> None:
        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2_data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            log.debug(
                f"No face detected in {image_path}. Returning the original image."
            )
            return

        # Use the first detected face
        (x, y, w, h) = faces[0]

        log.debug(f"Detected face at x={x}, y={y}, w={w}, h={h}")

        # Add padding around the face
        padding = int(padding_ratio * max(w, h))
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)

        log.debug(f"New coordinates x={x}, y={y}, w={w}, h={h}")

        # Crop and resize
        cropped_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).crop(
            (x, y, x + w, y + h)
        )
        cropped_image = cropped_image.resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
        cropped_image.save(image_path)

    def _write_training_images(
        self, lora_id: UUID4, content: UploadFile, resolution: int
    ) -> None:
        # unzip the contents of content.filename to a folder named lora_id
        assert content.filename is not None
        log.debug("writing training images", lora_id=lora_id, content=content.filename)
        lora_content_dir = f"{settings.CONTENT_DIR}/{lora_id}/content"
        os.makedirs(lora_content_dir, exist_ok=True)
        with ZipFile(content.file, "r") as zip_ref:
            for member in zip_ref.namelist():
                # skip directories and macosx files
                if member.endswith("/") or os.path.basename(member).startswith("."):
                    continue
                zip_ref.extract(member, lora_content_dir)
                self._format_image(
                    image_path=f"{lora_content_dir}/{member}",
                    resolution=resolution,
                )
                # move the image to lora_content_dir/basename
                os.rename(
                    f"{lora_content_dir}/{member}",
                    f"{lora_content_dir}/{os.path.basename(member)}",
                )
        # delete all non-files in lora_content_dir
        for root, dirs, files in os.walk(lora_content_dir):
            for dir in dirs:
                os.rmdir(f"{root}/{dir}")
        log.debug("training images written")

    def _train_lora(self, lora_id: UUID4, lora_create: LoraCreate) -> None:
        log.debug("loading training images...")
        write_basic_config(mixed_precision="bf16")
        instance_data_dir = f"{settings.CONTENT_DIR}/{lora_id}/content"
        instance_data_output_dir = f"{settings.CONTENT_DIR}/{lora_id}/results"
        os.makedirs(instance_data_dir, exist_ok=True)
        os.makedirs(instance_data_output_dir, exist_ok=True)
        log.debug("launching dreambooth training")
        assert lora_create.tuning_config.hf_hyperparameters is not None
        cmd = [
            "accelerate",
            "launch",
            "/root/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py",
            # half-precision floats most of the time for faster training
            "--mixed_precision=bf16",
            f"--pretrained_model_name_or_path={settings.MODEL_DIR}",
            f"--instance_data_dir={instance_data_dir}",
            f"--output_dir={instance_data_output_dir}",
            f"--rank={lora_create.tuning_config.hf_hyperparameters.rank}",
            f"--instance_prompt='{lora_create.tuning_config.training_prompt}'",
            f"--resolution={lora_create.tuning_config.hf_hyperparameters.resolution}",
            f"--train_batch_size={lora_create.tuning_config.hf_hyperparameters.train_batch_size}",
            f"--gradient_accumulation_steps={lora_create.tuning_config.hf_hyperparameters.gradient_accumulation_steps}",
            f"--learning_rate={lora_create.tuning_config.hf_hyperparameters.learning_rate}",
            f"--lr_scheduler={lora_create.tuning_config.hf_hyperparameters.lr_scheduler}",
            f"--lr_warmup_steps={lora_create.tuning_config.hf_hyperparameters.lr_warmup_steps}",
            f"--max_train_steps={lora_create.tuning_config.hf_hyperparameters.max_train_steps}",
            f"--checkpointing_steps={lora_create.tuning_config.hf_hyperparameters.checkpointing_steps}",
            f"--seed={lora_create.tuning_config.hf_hyperparameters.seed}",
        ]
        self._exec_subprocess(cmd)

    def _run_inference(self, photobooth_create: PhotoboothCreate) -> None:
        log.debug("loading model...")
        pipe = DiffusionPipeline.from_pretrained(
            settings.MODEL_DIR,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        log.debug("model loaded")
        log.debug("model type", model_type=type(pipe))
        log.debug("loading weights")
        results_dir = f"{settings.CONTENT_DIR}/{photobooth_create.lora_id}/results"
        # load the weights from the results directory
        log.debug("loading lora weights")
        pipe.load_lora_weights(
            results_dir,
            weight_name="pytorch_lora_weights.safetensors",
            local_files_only=True,
        )
        pipe.fuse_lora()
        log.debug("running inference...")
        images = pipe(
            prompt=photobooth_create.prompt,
            num_inference_steps=photobooth_create.inference_config.num_inference_steps,
            guidance_scale=photobooth_create.inference_config.guidance_scale,
            num_images_per_prompt=photobooth_create.inference_config.num_images_per_prompt,
            height=photobooth_create.inference_config.height,
            width=photobooth_create.inference_config.width,
        ).images
        # write each image to disk
        for image in images:
            image.save(f"{results_dir}/{uuid4()}.png")
        log.debug("inference complete")
