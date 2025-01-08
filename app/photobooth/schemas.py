import json
import random
from enum import Enum, unique

from pydantic import UUID4, BaseModel, field_validator, model_validator


@unique
class ModelName(str, Enum):
    flux_1_dev = "black-forest-labs/FLUX.1-dev"


@unique
class SafetensorsFilename(str, Enum):
    flux_1_dev = "flux1-dev.safetensors"


class HuggingfaceHyperparameters(BaseModel):
    seed: int = 117
    rank: int = 16
    resolution: int = 1024
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 2000
    checkpointing_steps: int = 2000

    @field_validator("seed", mode="before")
    def set_random_seed(cls, v: int) -> int:
        return random.randrange(1, 2**32) if v == 0 else v


class LoraTuningConfig(BaseModel):
    subject_name: str
    training_prompt: str
    hf_hyperparameters: HuggingfaceHyperparameters = HuggingfaceHyperparameters()


class LoraCreate(BaseModel):
    tuning_config: LoraTuningConfig

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value: dict) -> dict:
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class LoraCreateResponse(BaseModel):
    lora_id: UUID4
    detail: str


class InferenceConfig(BaseModel):
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    num_images_per_prompt: int = 1


class PhotoboothCreate(BaseModel):
    lora_id: UUID4
    prompt: str
    inference_config: InferenceConfig = InferenceConfig()
