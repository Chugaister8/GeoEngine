"""
GeoEngine — Terrain REST API
FastAPI router для DEM операцій.

Endpoints:
  GET  /api/terrain/sources
  GET  /api/terrain/tile/{z}/{x}/{y}/meta
  GET  /api/terrain/tile/{z}/{x}/{y}.png
  POST /api/terrain/elevation
  POST /api/terrain/mesh
  GET  /api/terrain/cache/stats
  DELETE /api/terrain/cache
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..services.terrain import TerrainService

router = APIRouter(prefix="/api/terrain", tags=["terrain"])

# ----------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ----------------------------------------------------------------

class ElevationRequest(BaseModel):
    points: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Список точок [[lat, lon], ...]",
    )
    source: str = "terrarium"
    zoom:   int = Field(default=11, ge=4, le=14)

    @field_validator("points")
    @classmethod
    def validate_points(cls, v: list[list[float]]) -> list[list[float]]:
        for pt in v:
            if len(pt) != 2:
                raise ValueError("Кожна точка має бути [lat, lon]")
            lat, lon = pt
            if not (-90 <= lat <= 90):
                raise ValueError(f"lat={lat} поза [-90, 90]")
            if not (-180 <= lon <= 180):
                raise ValueError(f"lon={lon} поза [-180, 180]")
        return v


class MeshRequest(BaseModel):
    west:         float = Field(..., ge=-180, le=180)
    south:        float = Field(..., ge=-90,  le=90)
    east:         float = Field(..., ge=-180, le=180)
    north:        float = Field(..., ge=-90,  le=90)
    source:       str   = "terrarium"
    max_vertices: int   = Field(default=65_536, ge=64, le=262_144)
    skirt_height_m: float = Field(default=200.0, ge=0.0)

    @field_validator("north")
    @classmethod
    def north_gt_south(cls, v: float, info: Any) -> float:
        data = info.data
        if "south" in data and v <= data["south"]:
            raise ValueError("north має бути > south")
        return v

    @property
    def area_deg2(self) -> float:
        return (self.east - self.west) * (self.north - self.south)


# ----------------------------------------------------------------
# DEPENDENCY
# ----------------------------------------------------------------

# TerrainService інжектується через FastAPI DI
# Реальний інстанс створюється у main.py (lifespan)
_terrain_service: TerrainService | None = None


def get_terrain_service() -> TerrainService:
    if _terrain_service is None:
        raise RuntimeError("TerrainService не ініціалізований")
    return _terrain_service


def set_terrain_service(svc: TerrainService) -> None:
    global _terrain_service
    _terrain_service = svc


# ----------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------

@router.get("/sources")
async def get_sources(
    svc: Annotated[TerrainService, Depends(get_terrain_service)],
) -> list[dict[str, Any]]:
    """Список доступних DEM джерел."""
    return await svc.get_sources()


@router.get("/tile/{z}/{x}/{y}/meta")
async def get_tile_meta(
    z:      int = Path(..., ge=0, le=22),
    x:      int = Path(..., ge=0),
    y:      int = Path(..., ge=0),
    source: str = Query(default="terrarium"),
    svc:    Annotated[TerrainService, Depends(get_terrain_service)] = None,
) -> dict[str, Any]:
    """Метадані тайлу (висоти, розмір, coverage)."""
    _validate_tile(x, y, z)
    try:
        return await svc.get_tile_meta(x=x, y=y, z=z, source=source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка сервера: {e}")


@router.get("/tile/{z}/{x}/{y}.png")
async def get_tile_png(
    z:        int = Path(..., ge=0, le=22),
    x:        int = Path(..., ge=0),
    y:        int = Path(..., ge=0),
    source:   str = Query(default="terrarium"),
    colormap: str = Query(default="terrain"),
    size:     int = Query(default=256, ge=64, le=512),
    svc:      Annotated[TerrainService, Depends(get_terrain_service)] = None,
) -> Response:
    """Тайл як PNG зображення."""
    _validate_tile(x, y, z)
    try:
        png_bytes = await svc.get_tile_png(
            x=x, y=y, z=z,
            source=source,
            colormap=colormap,
            size=size,
        )
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=86400",
                "X-Tile":        f"{z}/{x}/{y}",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/elevation")
async def get_elevation(
    request: ElevationRequest,
    svc:     Annotated[TerrainService, Depends(get_terrain_service)],
) -> dict[str, Any]:
    """
    Висоти для списку точок.

    Request body:
        {"points": [[lat, lon], ...], "source": "terrarium"}

    Response:
        {"points": [[lat, lon], ...], "elevations": [float|null, ...]}
    """
    try:
        elevations = await svc.get_elevations(
            points=[(pt[0], pt[1]) for pt in request.points],
            source=request.source,
            zoom=request.zoom,
        )
        return {
            "points":     request.points,
            "elevations": [
                round(float(e), 2) if e is not None else None
                for e in elevations
            ],
            "source":     request.source,
            "count":      len(elevations),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mesh")
async def get_mesh(
    request: MeshRequest,
    svc:     Annotated[TerrainService, Depends(get_terrain_service)],
) -> dict[str, Any]:
    """
    TerrainMesh для географічного bbox.

    Обмеження: bbox не більше 4°×4° (≈ 450×450 км).
    """
    # Перевірка розміру bbox
    if request.area_deg2 > 16.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"BBox занадто великий: {request.area_deg2:.1f}°². "
                "Максимум: 4°×4° (16°²)."
            ),
        )

    # Визначаємо центральний тайл
    from geoengine.geo.projection import latlon_to_tile
    from geoengine.geo.bbox import BBox

    center_lat = (request.south + request.north) / 2
    center_lon = (request.west  + request.east)  / 2

    # Zoom залежно від розміру bbox
    area   = request.area_deg2
    zoom   = 12 if area < 0.01 else 10 if area < 0.25 else 8

    tile   = latlon_to_tile(center_lat, center_lon, zoom=zoom)

    try:
        mesh_data = await svc.get_tile_mesh(
            x=tile.x,
            y=tile.y,
            z=tile.z,
            source=request.source,
            max_vertices=request.max_vertices,
            skirt_height_m=request.skirt_height_m,
        )
        return mesh_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def cache_stats(
    svc: Annotated[TerrainService, Depends(get_terrain_service)],
) -> dict[str, Any]:
    """Статистика кешу."""
    return await svc.cache_stats()


@router.delete("/cache")
async def clear_cache(
    source: str = Query(default="all"),
    svc:    Annotated[TerrainService, Depends(get_terrain_service)] = None,
) -> dict[str, Any]:
    """Очистити кеш."""
    return await svc.clear_cache(source=source)


# ----------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------

def _validate_tile(x: int, y: int, z: int) -> None:
    """Перевірити що тайл валідний для zoom рівня."""
    max_xy = (1 << z) - 1
    if x > max_xy or y > max_xy:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Невалідні координати тайлу: {z}/{x}/{y}. "
                f"При zoom={z} максимум: x,y ≤ {max_xy}"
            ),
)
