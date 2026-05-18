"""
GeoEngine — Simulation REST API
Endpoints для балістики та симуляції вогню.

Endpoints:
  POST /api/sim/ballistics     — розрахунок траєкторії
  POST /api/sim/ballistics/table — таблиця стрільби
  POST /api/sim/fire           — симуляція поширення вогню
  GET  /api/sim/ballistics/presets — доступні пресети снарядів
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from geoengine.simulation.ballistics import (
    BallisticsSolver, ProjectileParams,
    WindVector, LatLonAlt, BALLISTIC_PRESETS,
    optimal_elevation,
)
from geoengine.simulation.fire import FireSimulation

router = APIRouter(prefix="/api/sim", tags=["simulation"])


# ----------------------------------------------------------------
# МОДЕЛІ ЗАПИТІВ
# ----------------------------------------------------------------

class WindModel(BaseModel):
    speed_ms:     float = Field(default=0.0, ge=0.0, le=100.0)
    direction_deg: float = Field(default=0.0, ge=0.0, lt=360.0)
    vertical_ms:  float = Field(default=0.0)


class BallisticsRequest(BaseModel):
    # Точка пострілу
    lat:             float = Field(..., ge=-90, le=90)
    lon:             float = Field(..., ge=-180, le=180)
    alt_m:           float = Field(default=0.0)

    # Балістика
    azimuth_deg:      float = Field(..., ge=0.0, lt=360.0)
    elevation_deg:    float = Field(..., ge=0.1, le=89.9)
    muzzle_velocity:  float = Field(..., ge=10.0, le=2000.0)

    # Снаряд
    projectile_preset: str  = Field(default="artillery_122mm")
    projectile_mass:   float | None = None
    projectile_diam:   float | None = None
    projectile_cd:     float | None = None

    # Вітер
    wind: WindModel = Field(default_factory=WindModel)

    # Налаштування
    coriolis:      bool  = False
    max_time_s:    float = Field(default=300.0, le=600.0)
    include_geojson: bool = True


class BallisticsTableRequest(BaseModel):
    lat:             float = Field(..., ge=-90, le=90)
    lon:             float = Field(..., ge=-180, le=180)
    alt_m:           float = 0.0
    muzzle_velocity: float = Field(..., ge=10.0, le=2000.0)
    projectile_preset: str = "artillery_122mm"
    wind:            WindModel = Field(default_factory=WindModel)
    elevations_deg:  list[float] | None = None
    azimuths_deg:    list[float] | None = None


class FireRequest(BaseModel):
    # Точка займання
    lat:            float = Field(..., ge=-90, le=90)
    lon:            float = Field(..., ge=-180, le=180)

    # Параметри
    duration_hours: float = Field(default=6.0,  ge=0.1, le=48.0)
    radius_km:      float = Field(default=10.0, ge=1.0, le=50.0)
    cell_size_m:    float = Field(default=50.0, ge=10.0, le=500.0)

    # Метео
    wind:           WindModel = Field(default_factory=WindModel)
    moisture:       float = Field(default=0.06, ge=0.0, le=1.0)
    temperature_c:  float = Field(default=25.0, ge=-20.0, le=50.0)
    humidity_pct:   float = Field(default=40.0, ge=0.0, le=100.0)

    # Виходи
    n_frames:       int   = Field(default=12, ge=1, le=48)
    include_frames: bool  = True


# ----------------------------------------------------------------
# DEPENDENCY (TerrainService)
# ----------------------------------------------------------------

def get_terrain_service():
    from .terrain import get_terrain_service as _get
    return _get()


# ----------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------

@router.get("/ballistics/presets")
async def get_presets() -> dict[str, Any]:
    """Список доступних пресетів снарядів."""
    return {
        "presets": [
            {
                "id":       name,
                "mass_kg":  p["mass_kg"],
                "diameter_m": p["diameter_m"],
                "bc":       p.get("bc", 0.0),
                "cd":       p["cd"],
            }
            for name, p in BALLISTIC_PRESETS.items()
        ]
    }


@router.post("/ballistics")
async def calculate_ballistics(
    request: BallisticsRequest,
    svc:     Any = Depends(get_terrain_service),
) -> dict[str, Any]:
    """
    Розрахувати траєкторію снаряду.

    Враховує:
    - Опір повітря (drag)
    - Вітер
    - Реальний рельєф (якщо є DEM)
    - Ефект Коріоліса (опційно)
    """
    # Снаряд
    if (request.projectile_mass is not None and
            request.projectile_diam is not None):
        proj = ProjectileParams(
            mass_kg=request.projectile_mass,
            diameter_m=request.projectile_diam,
            cd=request.projectile_cd or 0.30,
        )
    else:
        try:
            proj = ProjectileParams.from_preset(request.projectile_preset)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Вітер
    wind = WindVector(
        speed_ms=request.wind.speed_ms,
        direction_deg=request.wind.direction_deg,
        vertical_ms=request.wind.vertical_ms,
    )

    # DEM тайл (завантажуємо якщо є сервіс)
    dem_tile = None
    if svc is not None:
        try:
            from geoengine.geo.projection import latlon_to_tile
            tile     = latlon_to_tile(request.lat, request.lon, zoom=10)
            dem_tile = await svc._get_dem_tile(
                tile.x, tile.y, tile.z, "terrarium"
            )
        except Exception:
            pass  # Без DEM теж працює

    solver = BallisticsSolver(
        dem_tile=dem_tile,
        max_time_s=request.max_time_s,
        coriolis=request.coriolis,
    )

    try:
        result = solver.solve(
            origin=LatLonAlt(
                lat=request.lat,
                lon=request.lon,
                alt=request.alt_m,
            ),
            azimuth_deg=request.azimuth_deg,
            elevation_deg=request.elevation_deg,
            muzzle_velocity=request.muzzle_velocity,
            projectile=proj,
            wind=wind,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = result.to_dict()
    if request.include_geojson:
        data["geojson"] = result.to_geojson()

    return data


@router.post("/ballistics/table")
async def ballistics_table(
    request: BallisticsTableRequest,
) -> dict[str, Any]:
    """
    Таблиця стрільби: дальності для різних кутів підвищення.
    """
    try:
        proj = ProjectileParams.from_preset(request.projectile_preset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    wind = WindVector(
        speed_ms=request.wind.speed_ms,
        direction_deg=request.wind.direction_deg,
    )

    solver = BallisticsSolver(dem_tile=None)

    table = solver.solve_range_table(
        origin=LatLonAlt(request.lat, request.lon, request.alt_m),
        muzzle_velocity=request.muzzle_velocity,
        projectile=proj,
        elevations_deg=request.elevations_deg,
        azimuths_deg=request.azimuths_deg,
        wind=wind,
    )

    return {
        "projectile":    request.projectile_preset,
        "muzzle_velocity": request.muzzle_velocity,
        "wind_speed_ms": request.wind.speed_ms,
        "entries":       table,
        "count":         len(table),
    }


@router.post("/fire")
async def simulate_fire(
    request: FireRequest,
    svc:     Any = Depends(get_terrain_service),
) -> dict[str, Any]:
    """
    Симуляція поширення лісового вогню.

    Модель: Rothermel (спрощена) + Cellular Automata.
    Повертає карту часу займання та анімаційні кадри.
    """
    # DEM тайл
    dem_tile = None
    if svc is not None:
        try:
            from geoengine.geo.projection import latlon_to_tile
            tile     = latlon_to_tile(request.lat, request.lon, zoom=10)
            dem_tile = await svc._get_dem_tile(
                tile.x, tile.y, tile.z, "terrarium"
            )
        except Exception:
            pass

    wind = WindVector(
        speed_ms=request.wind.speed_ms,
        direction_deg=request.wind.direction_deg,
    )

    sim = FireSimulation(
        dem_tile=dem_tile,
        wind=wind,
        cell_size_m=request.cell_size_m,
    )

    try:
        result = sim.run(
            ignition_lat=request.lat,
            ignition_lon=request.lon,
            duration_hours=request.duration_hours,
            radius_km=request.radius_km,
            moisture=request.moisture,
            temperature_c=request.temperature_c,
            humidity_pct=request.humidity_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = result.to_dict()

    if request.include_frames:
        data["frames"] = result.animation_frames(n_frames=request.n_frames)

    return data
