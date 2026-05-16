"""
GeoEngine — Terrain REST API
HTTP ендпоінти для роботи з терейном.

Маршрути:
  GET  /api/terrain/tile/{z}/{x}/{y}   — DEM тайл як PNG або JSON
  GET  /api/terrain/elevation          — висота точки
  POST /api/terrain/mesh               — mesh для bbox
  GET  /api/terrain/sources            — список доступних джерел
  GET  /api/terrain/stats              — статистика кешу

Також використовується для prefetch тайлів до WebSocket запитів.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

import numpy as np
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..services.terrain import TerrainService, get_terrain_service

log: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/terrain", tags=["terrain"])


# ----------------------------------------------------------------
# REQUEST / RESPONSE МОДЕЛІ
# ----------------------------------------------------------------

class ElevationRequest(BaseModel):
    """Запит висоти для списку точок."""
    points: list[tuple[float, float]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Список (lat, lon) точок"
    )
    source: str = "copernicus25"


class ElevationResponse(BaseModel):
    """Відповідь з висотами."""
    points: list[tuple[float, float]]
    elevations: list[float | None]   # None де дані відсутні
    source: str
    unit: str = "meters"


class MeshRequest(BaseModel):
    """Запит mesh для BBox."""
    west:           float = Field(..., ge=-180, le=180)
    south:          float = Field(..., ge=-90,  le=90)
    east:           float = Field(..., ge=-180, le=180)
    north:          float = Field(..., ge=-90,  le=90)
    source:         str   = "copernicus25"
    max_vertices:   int   = Field(default=65_536, ge=64, le=262_144)
    lod_level:      int   = Field(default=0, ge=0, le=5)


class MeshResponse(BaseModel):
    """Відповідь з mesh даними."""
    vertex_count:   int
    triangle_count: int
    bbox:           list[float]
    origin:         dict
    buffers:        dict   # base64 encoded
    min_elevation:  float
    max_elevation:  float
    resolution_m:   float
    memory_bytes:   int


class TileMetaResponse(BaseModel):
    """Метадані тайлу."""
    tile:           dict
    bbox:           list[float]
    min_elevation:  float
    max_elevation:  float
    mean_elevation: float
    coverage_pct:   float
    resolution_m:   float
    source:         str


class SourceInfo(BaseModel):
    """Інформація про DEM джерело."""
    id:              str
    name:            str
    resolution_m:    float
    global_coverage: bool
    requires_api_key: bool


class CacheStats(BaseModel):
    """Статистика кешу."""
    files:    int
    size_mb:  int
    sources:  dict[str, int]


# ----------------------------------------------------------------
# ЕНДПОІНТИ
# ----------------------------------------------------------------

@router.get("/sources", response_model=list[SourceInfo])
async def list_sources() -> list[SourceInfo]:
    """
    Список доступних DEM джерел.

    Returns:
        Список джерел з метаданими (resolution, coverage, тощо)
    """
    from ....packages.core_python.geoengine.dem.sources import SOURCES
    return [
        SourceInfo(
            id=str(src.id),
            name=src.name,
            resolution_m=src.resolution_m,
            global_coverage=src.global_coverage,
            requires_api_key=src.requires_api_key,
        )
        for src in SOURCES.values()
    ]


@router.get("/tile/{z}/{x}/{y}/meta", response_model=TileMetaResponse)
async def get_tile_meta(
    z:       int,
    x:       int,
    y:       int,
    source:  str = Query(default="copernicus25"),
    service: TerrainService = Depends(get_terrain_service),
) -> TileMetaResponse:
    """
    Метадані DEM тайлу без повної геометрії.
    Швидший ніж /mesh — для LOD планування.

    Args:
        z, x, y: XYZ адреса тайлу
        source:  DEM джерело

    Returns:
        Метадані: висоти, resolution, coverage, тощо
    """
    _validate_tile(z, x, y)

    try:
        meta = await service.get_tile_meta(
            tile_x=x, tile_y=y, tile_z=z, source=source,
        )
    except Exception as exc:
        log.error("api.tile_meta.error", tile=f"{z}/{x}/{y}", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TileMetaResponse(**meta)


@router.get(
    "/tile/{z}/{x}/{y}.png",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_tile_png(
    z:       int,
    x:       int,
    y:       int,
    source:  str   = Query(default="terrarium"),
    colormap: str  = Query(default="terrain"),   # terrain, grayscale, hillshade
    service: TerrainService = Depends(get_terrain_service),
) -> Response:
    """
    DEM тайл як PNG зображення.
    Підтримує різні colormaps для візуалізації.

    Args:
        z, x, y:  XYZ адреса
        source:   DEM джерело
        colormap: схема кольорів (terrain/grayscale/hillshade)

    Returns:
        PNG image (256×256 або 512×512)
    """
    _validate_tile(z, x, y)

    try:
        png_bytes = await service.get_tile_png(
            tile_x=x, tile_y=y, tile_z=z,
            source=source, colormap=colormap,
        )
    except Exception as exc:
        log.error("api.tile_png.error", tile=f"{z}/{x}/{y}", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",  # кешуємо 24г
            "X-Tile":        f"{z}/{x}/{y}",
            "X-Source":      source,
        },
    )


@router.post("/elevation", response_model=ElevationResponse)
async def get_elevation(
    request: ElevationRequest,
    service: TerrainService = Depends(get_terrain_service),
) -> ElevationResponse:
    """
    Висоти для списку точок.

    Пакетний запит: до 1000 точок за раз.
    Використовує біліарну інтерполяцію.

    Args:
        request: список (lat, lon) точок + джерело

    Returns:
        Список висот у метрах (None де дані відсутні)
    """
    try:
        elevations = await service.get_elevations(
            points=request.points,
            source=request.source,
        )
    except Exception as exc:
        log.error("api.elevation.error", count=len(request.points), error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ElevationResponse(
        points=request.points,
        elevations=elevations,
        source=request.source,
    )


@router.post("/mesh", response_model=MeshResponse)
async def get_mesh(
    request: MeshRequest,
    service: TerrainService = Depends(get_terrain_service),
) -> MeshResponse:
    """
    Terrain mesh для BBox.

    Повертає повний 3D меш у base64 буферах:
    vertices (Float32), indices (Uint32), uvs (Float32), normals (Float32).

    ⚠️ Для великих BBox може бути повільним — використовуй WebSocket
    для потокового підвантаження по тайлах.

    Args:
        request: BBox + source + параметри mesh

    Returns:
        MeshResponse з base64 буферами
    """
    # Перевірка розміру bbox
    area_deg2 = (request.east - request.west) * (request.north - request.south)
    if area_deg2 > 4.0:  # > ~2°×2° = ~50000км²
        raise HTTPException(
            status_code=400,
            detail=(
                f"BBox занадто великий ({area_deg2:.2f}°²). "
                "Максимум 4°². Використовуй WebSocket для великих областей."
            ),
        )

    try:
        from ....packages.core_python.geoengine.geo.bbox import BBox
        bbox = BBox(
            west=request.west, south=request.south,
            east=request.east, north=request.north,
        )
        mesh_data = await service.get_bbox_mesh(
            bbox=bbox,
            source=request.source,
            max_vertices=request.max_vertices,
            lod_level=request.lod_level,
        )
    except Exception as exc:
        log.error("api.mesh.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return MeshResponse(**mesh_data)


@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats(
    service: TerrainService = Depends(get_terrain_service),
) -> CacheStats:
    """Статистика DEM кешу."""
    stats = await service.get_cache_stats()
    return CacheStats(**stats)


@router.delete("/cache")
async def clear_cache(
    source: str | None = Query(default=None),
    service: TerrainService = Depends(get_terrain_service),
) -> dict:
    """
    Очистити DEM кеш.

    Args:
        source: очистити конкретне джерело або все (None)

    Returns:
        {"deleted": N} — кількість видалених файлів
    """
    deleted = await service.clear_cache(source=source)
    return {"deleted": deleted, "source": source or "all"}


# ----------------------------------------------------------------
# УТИЛІТИ
# ----------------------------------------------------------------

def _validate_tile(z: int, x: int, y: int) -> None:
    """Валідувати XYZ тайл координати."""
    if not (0 <= z <= 22):
        raise HTTPException(status_code=400, detail=f"zoom={z} поза [0, 22]")
    max_idx = (1 << z) - 1
    if not (0 <= x <= max_idx):
        raise HTTPException(status_code=400, detail=f"x={x} поза [0, {max_idx}]")
    if not (0 <= y <= max_idx):
        raise HTTPException(status_code=400, detail=f"y={y} поза [0, {max_idx}]")
