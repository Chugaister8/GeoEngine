"""
GeoEngine — Pytest Fixtures
Спільні fixtures для всіх Python тестів.

Структура:
  conftest.py      — глобальні fixtures
  test_bbox.py     — BBox тести
  test_coords.py   — Coordinate тести
  test_projection.py — Tile/projection тести
  test_dem.py      — DEM loader/processor тести
  test_mesh.py     — Mesh builder тести
  test_analysis.py — GIS analysis тести
  test_osm.py      — OSM fetcher/parser тести
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

# ----------------------------------------------------------------
# КОНСТАНТИ ДЛЯ ТЕСТІВ
# ----------------------------------------------------------------

# Тестові BBox (реальні регіони)
BBOX_CARPATHIANS = (22.0, 47.5, 25.0, 49.5)   # Карпати (W, S, E, N)
BBOX_KYIV        = (30.3, 50.3, 30.7, 50.6)   # Київ
BBOX_SMALL       = (23.0, 48.0, 23.1, 48.1)   # маленький (0.1°×0.1°)
BBOX_TINY        = (23.0, 48.0, 23.01, 48.01) # дуже маленький

# Тестові координати
POINT_KYIV       = (50.45, 30.52)    # (lat, lon)
POINT_LVIV       = (49.84, 24.02)
POINT_HOVERLA    = (48.16, 24.50)    # найвища точка України
POINT_ZERO       = (0.0, 0.0)
POINT_ANTIMERIDIAN = (0.0, 179.99)


# ----------------------------------------------------------------
# СИНТЕТИЧНИЙ DEM (без мережевих запитів)
# ----------------------------------------------------------------

def make_synthetic_dem(
    width:  int   = 64,
    height: int   = 64,
    min_h:  float = 200.0,
    max_h:  float = 2061.0,  # висота Говерли
    seed:   int   = 42,
) -> np.ndarray:
    """
    Генерувати синтетичний heightmap для тестів.
    Використовує суму синусоїд для правдоподібного рельєфу.
    """
    rng = np.random.default_rng(seed)
    x   = np.linspace(0, 4 * np.pi, width)
    y   = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Базовий рельєф (гірська місцевість)
    terrain = (
        np.sin(xx) * np.cos(yy) * 0.4
        + np.sin(xx * 2.3) * np.cos(yy * 1.7) * 0.3
        + np.sin(xx * 5.1) * np.cos(yy * 4.3) * 0.15
        + rng.random((height, width)) * 0.15  # шум
    )

    # Нормалізуємо до [min_h, max_h]
    t_min, t_max = terrain.min(), terrain.max()
    terrain = (terrain - t_min) / (t_max - t_min)
    terrain = terrain * (max_h - min_h) + min_h

    return terrain.astype(np.float32)


# ----------------------------------------------------------------
# FIXTURES — BBox
# ----------------------------------------------------------------

@pytest.fixture
def bbox_carpathians():
    """BBox Карпат."""
    from geoengine.geo.bbox import BBox
    w, s, e, n = BBOX_CARPATHIANS
    return BBox(west=w, south=s, east=e, north=n)


@pytest.fixture
def bbox_kyiv():
    """BBox Києва."""
    from geoengine.geo.bbox import BBox
    w, s, e, n = BBOX_KYIV
    return BBox(west=w, south=s, east=e, north=n)


@pytest.fixture
def bbox_small():
    """Маленький BBox для швидких тестів."""
    from geoengine.geo.bbox import BBox
    w, s, e, n = BBOX_SMALL
    return BBox(west=w, south=s, east=e, north=n)


# ----------------------------------------------------------------
# FIXTURES — DEM
# ----------------------------------------------------------------

@pytest.fixture
def synthetic_heightmap():
    """Синтетичний heightmap 64×64."""
    return make_synthetic_dem(64, 64)


@pytest.fixture
def synthetic_dem_tile(bbox_small):
    """Синтетичний DEMTile для тестів."""
    from rasterio.transform import from_bounds
    from geoengine.dem.loader import DEMTile

    data = make_synthetic_dem(64, 64)
    bbox = bbox_small

    transform = from_bounds(
        bbox.west, bbox.south, bbox.east, bbox.north,
        data.shape[1], data.shape[0],
    )

    return DEMTile(
        data=data,
        bbox=bbox,
        transform=transform,
        crs="EPSG:4326",
        source="synthetic",
        nodata=-9999.0,
    )


@pytest.fixture
def dem_tile_with_nodata(bbox_small):
    """DEMTile з NaN значеннями."""
    from rasterio.transform import from_bounds
    from geoengine.dem.loader import DEMTile

    data = make_synthetic_dem(32, 32)

    # Додаємо NaN в декількох місцях
    data[0:5, 0:5]   = np.nan
    data[10:15, 8:12] = np.nan

    bbox = bbox_small
    transform = from_bounds(
        bbox.west, bbox.south, bbox.east, bbox.north,
        data.shape[1], data.shape[0],
    )

    return DEMTile(
        data=data,
        bbox=bbox,
        transform=transform,
        crs="EPSG:4326",
        source="synthetic_nodata",
        nodata=-9999.0,
    )


@pytest.fixture
def tmp_geotiff(synthetic_dem_tile, tmp_path):
    """
    Тимчасовий GeoTIFF файл для тестів DEMLoader.
    Автоматично видаляється після тесту.
    """
    import rasterio
    from rasterio.transform import from_bounds

    path  = tmp_path / "test_dem.tif"
    tile  = synthetic_dem_tile
    h, w  = tile.data.shape

    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=tile.transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(tile.data, 1)

    return path


# ----------------------------------------------------------------
# FIXTURES — OSM
# ----------------------------------------------------------------

@pytest.fixture
def osm_overpass_response():
    """Мок-відповідь Overpass API з будівлями та дорогами."""
    return {
        "elements": [
            # Вузли
            {"type": "node", "id": 1, "lat": 48.001, "lon": 23.001},
            {"type": "node", "id": 2, "lat": 48.002, "lon": 23.001},
            {"type": "node", "id": 3, "lat": 48.002, "lon": 23.002},
            {"type": "node", "id": 4, "lat": 48.001, "lon": 23.002},
            {"type": "node", "id": 5, "lat": 48.003, "lon": 23.003},
            {"type": "node", "id": 6, "lat": 48.004, "lon": 23.004},
            {"type": "node", "id": 7, "lat": 48.005, "lon": 23.005},
            # Будівля (замкнений полігон)
            {
                "type": "way",
                "id": 100,
                "nodes": [1, 2, 3, 4, 1],
                "tags": {
                    "building": "residential",
                    "height":   "9",
                    "name":     "Test Building",
                },
            },
            # Дорога
            {
                "type": "way",
                "id": 200,
                "nodes": [5, 6, 7],
                "tags": {
                    "highway": "primary",
                    "name":    "Test Road",
                },
            },
            # Будівля без explicit висоти
            {
                "type": "way",
                "id": 101,
                "nodes": [2, 3, 6, 5, 2],
                "tags": {
                    "building":        "apartments",
                    "building:levels": "5",
                },
            },
        ]
    }


@pytest.fixture
def osm_data(osm_overpass_response, bbox_small):
    """OSMData розпарсений з мок-відповіді."""
    from geoengine.osm.parser import parse_overpass_json
    return parse_overpass_json(osm_overpass_response, bbox_small)


# ----------------------------------------------------------------
# FIXTURES — SERVER (FastAPI)
# ----------------------------------------------------------------

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def test_client():
    """Async HTTP клієнт для тестування FastAPI."""
    from httpx import AsyncClient, ASGITransport
    from apps.server.src.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def ws_client():
    """WebSocket тест-клієнт."""
    from fastapi.testclient import TestClient
    from apps.server.src.main import app

    with TestClient(app) as client:
        yield client


# ----------------------------------------------------------------
# FIXTURES — MOCK SERVICES
# ----------------------------------------------------------------

@pytest.fixture
def mock_dem_source_manager():
    """Мок DEMSourceManager без реальних мережевих запитів."""
    with patch(
        "geoengine.dem.sources.DEMSourceManager.fetch",
    ) as mock_fetch:
        mock_fetch.return_value = AsyncMock()
        yield mock_fetch


@pytest.fixture
def mock_overpass():
    """Мок Overpass API."""
    with patch(
        "geoengine.osm.fetcher.OverpassFetcher._fetch_with_retry",
    ) as mock:
        yield mock
