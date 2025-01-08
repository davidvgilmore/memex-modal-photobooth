import os
from enum import Enum
from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    test = "test"
    local = "local"
    preview = "preview"
    production = "production"


environment = Environment(os.getenv("APP_ENV", Environment.local.value))
environment_file = f".env.{environment.value}"


class Settings(BaseSettings):
    ENV: Environment = Environment.local
    DEBUG: bool = False
    LOG_LEVEL: str = "DEBUG"

    MODEL_DIR: str = "tmp/models"
    CONTENT_DIR: str = "tmp/content"

    # secret key
    SECRET_KEY: str = "thisisnotsecret"

    # CORS
    CORS_ORIGINS: str = "127.0.0.1"

    # HF
    HF_TOKEN: str = ""
    HF_HUB_ENABLE_HF_TRANSFER: int = 1

    # sqlite
    SQLITE_SCHEME: str = "sqlite+aiosqlite"
    SQLITE_DATABASE: str = "photobooth"
    SQLITE_PATH: str = "sqlite.db"
    SQLITE_USER: str = "sqlite"
    SQLITE_PWD: str = "sqlite"
    SQLITE_POOL_SIZE: int = 5
    SQLITE_POOL_RECYCLE_SECONDS: int = 600  # 10 minutes

    model_config = SettingsConfigDict(
        env_prefix="app_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_file=environment_file,
        extra="allow",
    )

    @cached_property
    def sqlite_dsn(self) -> str:
        return f"{self.SQLITE_SCHEME}:///{self.SQLITE_PATH}/{self.SQLITE_DATABASE}"

    def is_environment(self, environment: Environment) -> bool:
        return self.ENV == environment

    def is_test(self) -> bool:
        return self.is_environment(Environment.test)

    def is_local(self) -> bool:
        return self.is_environment(Environment.local)

    def is_preview(self) -> bool:
        return self.is_environment(Environment.preview)

    def is_production(self) -> bool:
        return self.is_environment(Environment.production)


settings = Settings()
