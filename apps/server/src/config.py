"""
GeoEngine — Server Configuration
Всі налаштування через Pydantic Settings + .env файл.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Налаштування сервера.

    Завантажуються з:
    1. .env файл (GEOENGINE_.env)
    2. Змінні середовища (GEOENGINE_*)
    3. Дефолтні значення
    """

    model_config = SettingsConfigDict(
        env_prefix="GEOENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---- Server ----
    host:      str   = "0.0.0.0"
    port:      int   = Field(default=8000, ge=1, le=65535)
    debug:     bool  = False
    log_level: str   = "info"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ---- WebSocket ----
    ws_timeout_s:        float = 60.0    # таймаут keepalive
    ws_max_message_size: int   = 10 * 1024 * 1024  # 10MB

    # ---- DEM / Terrain ----
    dem_cache_dir:    Path  = Path.home() / ".geoengine" / "dem_cache"
    dem_api_keys:     dict[str, str] = Field(default_factory=dict)
    terrain_workers:  int   = 4    # ThreadPool для mesh build
    dem_fetch_workers: int  = 4    # паралельні DEM завантаження
    mesh_cache_size:  int   = 256  # max кешованих mesh dict-ів

    # ---- Analysis ----
    analysis_workers: int = 2

    # ---- Paths ----
    data_dir: Path = Path("./data")

    @field_validator("dem_cache_dir", "data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        return Path(v).expanduser()

    @field_validator("dem_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> dict[str, str]:
        """Підтримує як dict так і JSON рядок."""
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton Settings — кешується після першого виклику."""
    return Settings()


# Зручний доступ
settings = get_settings()
