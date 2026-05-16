"""
GeoEngine — DEM Sources
Завантаження висотних даних з відкритих онлайн джерел.

Джерела:
  - OpenTopography API (SRTM30, SRTM90, Copernicus)
  - SRTM через tile URL (NASA)
  - Mapzen/Terrarium tiles (AWS)
  - Локальний кеш
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final
from urllib.parse import urlencode

import httpx
import numpy as np
import numpy.typing as npt
import structlog

from ..geo.bbox import BBox
from ..geo.projection import TileXYZ, tile_to_bbox, bbox_to_tiles, tile_resolution_m
from .loader import DEMLoader, DEMTile, DEMLoadError

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------------

CACHE_DIR_DEFAULT: Final[Path] = Path.home() / ".geoengine" / "dem_cache"
REQUEST_TIMEOUT:   Final[float] = 60.0   # секунд
MAX_RETRIES:       Final[int]   = 3
CHUNK_SIZE:        Final[int]   = 1024 * 256  # 256 KB


# ----------------------------------------------------------------
# ENUM ДЖЕРЕЛ
# ----------------------------------------------------------------

class DEMSourceID(StrEnum):
    """Ідентифікатори підтримуваних DEM джерел."""
    SRTM30       = "srtm30"       # NASA SRTM 30m (глобально)
    SRTM90       = "srtm90"       # NASA SRTM 90m (глобально)
    COPERNICUS25 = "copernicus25" # ESA Copernicus GLO-30 (25m)
    TERRARIUM    = "terrarium"    # Mapzen Terrarium tiles (AWS)
    CUSTOM       = "custom"       # Локальний файл


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Конфігурація одного DEM джерела."""
    id:           DEMSourceID
    name:         str
    resolution_m: float
    global_coverage: bool
    url_template: str          # {z}, {x}, {y}, {bbox_*} placeholders
    requires_api_key: bool = False
    tile_format:  str = "tif"  # tif | png | hgt


# ----------------------------------------------------------------
# РЕЄСТР ДЖЕРЕЛ
# ----------------------------------------------------------------

SOURCES: dict[DEMSourceID, SourceConfig] = {

    DEMSourceID.SRTM30: SourceConfig(
        id=DEMSourceID.SRTM30,
        name="NASA SRTM 30m",
        resolution_m=30.0,
        global_coverage=True,
        # OpenTopography безкоштовний API (потрібна реєстрація для >100 запитів)
        url_template=(
            "https://portal.opentopography.org/API/globaldem"
            "?demtype=SRTMGL1"
            "&south={bbox_south}&north={bbox_north}"
            "&west={bbox_west}&east={bbox_east}"
            "&outputFormat=GTiff"
            "&API_Key={api_key}"
        ),
        requires_api_key=True,
    ),

    DEMSourceID.SRTM90: SourceConfig(
        id=DEMSourceID.SRTM90,
        name="NASA SRTM 90m",
        resolution_m=90.0,
        global_coverage=True,
        url_template=(
            "https://portal.opentopography.org/API/globaldem"
            "?demtype=SRTMGL3"
            "&south={bbox_south}&north={bbox_north}"
            "&west={bbox_west}&east={bbox_east}"
            "&outputFormat=GTiff"
            "&API_Key={api_key}"
        ),
        requires_api_key=True,
    ),

    DEMSourceID.COPERNICUS25: SourceConfig(
        id=DEMSourceID.COPERNICUS25,
        name="ESA Copernicus DEM GLO-30 (25m)",
        resolution_m=25.0,
        global_coverage=True,
        url_template=(
            "https://portal.opentopography.org/API/globaldem"
            "?demtype=COP30"
            "&south={bbox_south}&north={bbox_north}"
            "&west={bbox_west}&east={bbox_east}"
            "&outputFormat=GTiff"
            "&API_Key={api_key}"
        ),
        requires_api_key=True,
    ),

    DEMSourceID.TERRARIUM: SourceConfig(
        id=DEMSourceID.TERRARIUM,
        name="Mapzen Terrarium (AWS, безкоштовно)",
        resolution_m=30.0,  # залежить від zoom
        global_coverage=True,
        # RGB encoded elevation tiles
        # Висота = (R * 256 + G + B / 256) - 32768
        url_template=(
            "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
            "/{z}/{x}/{y}.png"
        ),
        requires_api_key=False,
        tile_format="png",
    ),
}


# ----------------------------------------------------------------
# DEM SOURCE MANAGER
# ----------------------------------------------------------------

class DEMSourceManager:
    """
    Менеджер завантаження DEM даних з онлайн джерел.

    Підтримує:
    - Async завантаження (httpx)
    - Дисковий LRU кеш
    - Автоматичні повтори при помилках
    - Паралельне завантаження тайлів

    Usage:
        manager = DEMSourceManager(cache_dir="~/.geoengine/cache")
        tile = await manager.fetch(
            bbox=BBox(22, 47, 24, 49),
            source=DEMSourceID.COPERNICUS25,
        )
    """

    def __init__(
        self,
        cache_dir:     str | Path = CACHE_DIR_DEFAULT,
        api_keys:      dict[str, str] | None = None,
        max_workers:   int = 4,
    ) -> None:
        """
        Args:
            cache_dir:   директорія для кешування завантажених файлів
            api_keys:    словник {source_id: api_key}
            max_workers: кількість паралельних завантажень
        """
        self._cache_dir   = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_keys    = api_keys or {}
        self._max_workers = max_workers
        self._loader      = DEMLoader()
        self._semaphore   = asyncio.Semaphore(max_workers)

    # ---- Публічний API ----

    async def fetch(
        self,
        bbox:    BBox,
        source:  DEMSourceID = DEMSourceID.COPERNICUS25,
    ) -> DEMTile:
        """
        Завантажити DEM для bbox з вказаного джерела.

        Алгоритм:
        1. Перевірити кеш
        2. Завантажити якщо немає
        3. Зберегти в кеш
        4. Повернути DEMTile

        Args:
            bbox:   географічний BBox
            source: ідентифікатор джерела

        Returns:
            DEMTile з даними висот

        Raises:
            DEMLoadError:      помилка завантаження
            DEMSourceAPIError: помилка API (401, 429, 500, ...)
        """
        cache_key  = self._cache_key(bbox, source)
        cache_path = self._cache_dir / source / f"{cache_key}.tif"

        # Спробувати кеш
        if cache_path.exists():
            log.debug("dem.fetch.cache_hit", source=source, cache_key=cache_key)
            return self._loader.load(cache_path, bbox=bbox)

        log.info("dem.fetch.download", source=source, bbox=str(bbox))

        # Завантажити
        config = SOURCES.get(source)
        if config is None:
            raise DEMLoadError(f"Невідоме джерело: {source}")

        raw_bytes = await self._download(bbox, config)

        # Зберегти в кеш
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw_bytes)

        return self._loader.load(cache_path, bbox=bbox)

    async def fetch_tiles(
        self,
        bbox:   BBox,
        zoom:   int,
        source: DEMSourceID = DEMSourceID.TERRARIUM,
    ) -> list[DEMTile]:
        """
        Завантажити DEM як XYZ тайли (для Terrarium / тайлових джерел).
        Паралельно завантажує всі тайли в bbox на заданому zoom.

        Args:
            bbox:   географічний BBox
            zoom:   рівень зуму (7-12 рекомендовано для DEM)
            source: джерело (рекомендовано TERRARIUM)

        Returns:
            Список DEMTile, по одному на кожен XYZ тайл
        """
        tiles = bbox_to_tiles(bbox, zoom)
        log.info(
            "dem.fetch_tiles.start",
            count=len(tiles), zoom=zoom, source=source,
        )

        tasks = [self._fetch_single_tile(t, source) for t in tiles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        dem_tiles: list[DEMTile] = []
        for tile, result in zip(tiles, results, strict=True):
            if isinstance(result, Exception):
                log.warning(
                    "dem.fetch_tiles.error",
                    tile=str(tile), error=str(result),
                )
            else:
                dem_tiles.append(result)

        log.info("dem.fetch_tiles.done", loaded=len(dem_tiles), total=len(tiles))
        return dem_tiles

    # ---- Кеш ----

    def cache_info(self) -> dict[str, int]:
        """Інформація про кеш: кількість файлів та розмір."""
        total_files = 0
        total_bytes = 0
        for f in self._cache_dir.rglob("*.tif"):
            total_files += 1
            total_bytes += f.stat().st_size
        return {
            "files": total_files,
            "size_mb": total_bytes // (1024 * 1024),
        }

    def clear_cache(self, source: DEMSourceID | None = None) -> int:
        """
        Очистити кеш.

        Args:
            source: очистити тільки це джерело, або все якщо None

        Returns:
            Кількість видалених файлів
        """
        target = self._cache_dir / source if source else self._cache_dir
        count = 0
        for f in target.rglob("*.tif"):
            f.unlink()
            count += 1
        log.info("dem.cache.cleared", files=count, source=source)
        return count

    # ---- Приватні методи ----

    async def _download(self, bbox: BBox, config: SourceConfig) -> bytes:
        """Завантажити DEM як bytes."""
        if config.tile_format == "png":
            raise DEMLoadError(
                f"Джерело {config.id} вимагає тайлового завантаження. "
                "Використовуй fetch_tiles()."
            )

        api_key = self._api_keys.get(config.id, "")
        if config.requires_api_key and not api_key:
            raise DEMSourceAPIError(
                f"API ключ не вказаний для {config.name}. "
                f"Передай api_keys={{'{config.id}': 'your_key'}} в DEMSourceManager."
            )

        url = config.url_template.format(
            bbox_west=bbox.west,
            bbox_south=bbox.south,
            bbox_east=bbox.east,
            bbox_north=bbox.north,
            api_key=api_key,
        )

        return await self._http_get_bytes(url)

    async def _fetch_single_tile(
        self,
        tile:   TileXYZ,
        source: DEMSourceID,
    ) -> DEMTile:
        """Завантажити один XYZ тайл."""
        async with self._semaphore:
            config = SOURCES[source]
            cache_key  = f"{tile.z}_{tile.x}_{tile.y}"
            cache_path = self._cache_dir / source / f"{cache_key}.png"

            if not cache_path.exists():
                url = config.url_template.format(
                    z=tile.z, x=tile.x, y=tile.y
                )
                raw = await self._http_get_bytes(url)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(raw)

            # Для Terrarium PNG: декодуємо RGB → висота
            if config.tile_format == "png":
                return self._decode_terrarium_tile(cache_path, tile)

            return self._loader.load_for_tile(cache_path, tile)

    @staticmethod
    def _decode_terrarium_tile(path: Path, tile: TileXYZ) -> DEMTile:
        """
        Декодувати Mapzen Terrarium PNG тайл у висоти.

        Формула: elevation = (R * 256 + G + B / 256) - 32768
        Де R, G, B — канали PNG зображення (0-255).
        """
        from PIL import Image
        from rasterio.transform import from_bounds

        img  = Image.open(path).convert("RGB")
        arr  = np.array(img, dtype=np.float32)

        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        elevation = (R * 256.0 + G + B / 256.0) - 32768.0

        # Нульові пікселі (0,0,0) → nodata (океан або відсутні дані)
        nodata_mask = (R == 0) & (G == 0) & (B == 0)
        elevation[nodata_mask] = np.nan

        bbox = tile_to_bbox(tile)
        h, w = elevation.shape
        transform = from_bounds(
            bbox.west, bbox.south, bbox.east, bbox.north, w, h
        )

        return DEMTile(
            data=elevation,
            bbox=bbox,
            transform=transform,
            crs="EPSG:4326",
            source=str(path),
            nodata=-32768.0,
        )

    async def _http_get_bytes(self, url: str) -> bytes:
        """HTTP GET з retry логікою."""
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    log.debug("dem.http.get", url=url[:80], attempt=attempt)
                    response = await client.get(url)

                    if response.status_code == 200:
                        return response.content

                    if response.status_code == 401:
                        raise DEMSourceAPIError(
                            f"Невірний API ключ (401): {url[:60]}"
                        )
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        log.warning("dem.http.rate_limit", wait=wait)
                        await asyncio.sleep(wait)
                        continue
                    if response.status_code >= 500:
                        raise DEMSourceAPIError(
                            f"Серверна помилка {response.status_code}: {url[:60]}"
                        )

                    raise DEMLoadError(
                        f"HTTP {response.status_code}: {url[:60]}"
                    )

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES:
                        wait = 2 ** attempt
                        log.warning(
                            "dem.http.retry",
                            attempt=attempt, wait=wait, error=str(exc),
                        )
                        await asyncio.sleep(wait)

        raise DEMLoadError(
            f"Не вдалося завантажити після {MAX_RETRIES} спроб: {last_error}"
        )

    @staticmethod
    def _cache_key(bbox: BBox, source: DEMSourceID) -> str:
        """Детермінований ключ кешу для bbox + source."""
        raw = f"{source}_{bbox.west:.6f}_{bbox.south:.6f}_{bbox.east:.6f}_{bbox.north:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ----------------------------------------------------------------
# EXCEPTIONS
# ----------------------------------------------------------------

class DEMSourceAPIError(DEMLoadError):
    """Помилка зовнішнього API (auth, rate limit, server error)."""
    pass
