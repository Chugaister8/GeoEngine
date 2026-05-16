"""
GeoEngine — Base Configuration
Базовий Pydantic Settings клас для всіх компонентів.

Кожен компонент (server, cli, jupyter) наслідує BaseGeoConfig
та додає власні поля.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseGeoConfig(BaseSettings):
    """
    Базова конфігурація GeoEngine.
    Читає змінні середовища з префіксом GEOENGINE_.

    Поля:
        log_level:    рівень логування
        debug:        режим відлагодження
        cache_dir:    директорія кешу
        data_dir:     директорія даних
        dem_api_keys: API ключі для DEM джерел
    """

    model_config = SettingsConfigDict(
        env_prefix="GEOENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Загальні ----
    log_level:   str  = Field(default="INFO",  description="Рівень логування")
    debug:       bool = Field(default=False,   description="Режим відлагодження")
    json_logs:   bool = Field(default=False,   description="JSON логи (для prod)")

    # ---- Директорії ----
    cache_dir: Path = Field(
        default=Path.home() / ".geoengine" / "cache",
        description="Директорія кешу",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Директорія даних",
    )

    # ---- DEM API ключі ----
    dem_api_keys: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "API ключі для DEM джерел.\n"
            "Формат: {source_id: api_key}\n"
            "Наприклад: {'copernicus25': 'your_key'}"
        ),
    )

    # ---- Продуктивність ----
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Максимальна кількість воркерів ThreadPool",
    )

    @field_validator("cache_dir", "data_dir", mode="before")
    @classmethod
    def expand_and_create(cls, v: Any) -> Path:
        """Розгортаємо ~ та створюємо директорію якщо не існує."""
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("dem_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> dict[str, str]:
        """Підтримує JSON рядок або dict."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_level(cls, v: Any) -> str:
        return str(v).upper()

    def setup_logging(self) -> None:
        """Ініціалізувати логування з цією конфігурацією."""
        from .logging import configure_logging
        configure_logging(
            level=self.log_level,
            json_output=self.json_logs,
        )

    def dem_cache_dir(self, source: str = "") -> Path:
        """Директорія кешу для конкретного DEM джерела."""
        path = self.cache_dir / "dem" / source if source else self.cache_dir / "dem"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tile_cache_dir(self) -> Path:
        """Директорія кешу тайлів."""
        path = self.cache_dir / "tiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mesh_cache_dir(self) -> Path:
        """Директорія кешу мешів."""
        path = self.cache_dir / "meshes"
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---- Глобальна конфігурація (lazy singleton) ----

_config: BaseGeoConfig | None = None

def get_config() -> BaseGeoConfig:
    """Отримати глобальну конфігурацію (singleton)."""
    global _config
    if _config is None:
        _config = BaseGeoConfig()
    return _config
