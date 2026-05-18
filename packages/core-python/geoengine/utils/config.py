"""
GeoEngine — Configuration
Pydantic Settings для всієї конфігурації.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseGeoConfig(BaseSettings):
    """
    Базова конфігурація GeoEngine.
    Всі параметри можна перевизначити через .env або змінні середовища.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GEOENGINE_",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Server ----
    host:       str   = Field(default="0.0.0.0",   description="Bind host")
    port:       int   = Field(default=8000,         description="HTTP port")
    debug:      bool  = Field(default=False,        description="Debug mode")
    log_level:  str   = Field(default="INFO",       description="Log level")
    json_logs:  bool  = Field(default=False,        description="JSON logs")

    # ---- Cache ----
    dem_cache_dir:  str = Field(
        default="~/.geoengine/dem_cache",
        description="DEM tile cache directory",
    )
    osm_cache_dir:  str = Field(
        default="~/.geoengine/osm_cache",
        description="OSM cache directory",
    )
    redis_url:      str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL",
    )

    # ---- API Keys ----
    dem_api_keys:   dict[str, str] = Field(
        default_factory=dict,
        description="DEM source API keys {source: key}",
    )

    # ---- CORS ----
    cors_origins:   list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # ---- Workers ----
    terrain_workers:  int = Field(default=4)
    analysis_workers: int = Field(default=2)

    @field_validator("dem_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> dict[str, str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v or ["http://localhost:3000"]

    @property
    def dem_cache_path(self) -> Path:
        return Path(self.dem_cache_dir).expanduser()

    @property
    def osm_cache_path(self) -> Path:
        return Path(self.osm_cache_dir).expanduser()
