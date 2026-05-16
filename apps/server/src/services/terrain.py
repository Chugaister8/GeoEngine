"""
GeoEngine — Terrain Service
Бізнес-логіка між API/WebSocket та core-python пакетами.

Відповідає за:
- Координацію DEM завантаження + mesh побудови
- Async виконання важких операцій у ThreadPool
- Кешування результатів (пам'ять + диск)
- Перетворення внутрішніх типів у wire-формат (dict для JSON/WS)
"""

from __future__ import annotations

import asyncio
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
import structlog

from ..config import settings

log: structlog.BoundLogger = structlog.get_logger(__name__)

# Пул потоків для CPU-bound операцій (numpy, mesh build)
_thread_pool = ThreadPoolExecutor(
    max_workers=settings.terrain_workers,
    thread_name_prefix="geo_terrain",
)


class TerrainService:
    """
    Сервіс терейну.

    Всі публічні методи — async.
    CPU-важкі операції делегуються у ThreadPoolExecutor
    через asyncio.get_event_loop().run_in_executor().
    """

    def __init__(self) -> None:
        # Імпортуємо core-python пакети
        from geoengine.dem.loader import DEMLoader
        from geoengine.dem.sources import DEMSourceManager, DEMSourceID
        from geoengine.mesh.terrain import TerrainMeshBuilder
        from geoengine.mesh.lod import LODManager, DEFAULT_LOD_PYRAMID
        from geoengine.geo.coords import LLH

        self._loader   = DEMLoader()
        self._source_mgr = DEMSourceManager(
            cache_dir=settings.dem_cache_dir,
            api_keys=settings.dem_api_keys,
            max_workers=settings.dem_fetch_workers,
        )
        self._mesh_builder = TerrainMeshBuilder(skirt_height=200.0)
        self._LLH = LLH
        self._DEMSourceID = DEMSourceID

        # In-memory кеш готових mesh dict-ів (LRU через functools)
        self._mesh_cache: dict[str, dict] = {}
        self._max_cache   = settings.mesh_cache_size

    # ---- Публічний API ----

    async def get_tile_mesh(
        self,
        tile_x:        int,
        tile_y:        int,
        tile_z:        int,
        source:        str = "copernicus25",
        max_vertices:  int = 65_536,
        skirt_height_m: float = 200.0,
    ) -> dict:
        """
        Отримати mesh тайлу як wire-format dict.

        Кеш-стратегія:
          L1: in-memory dict cache (швидко)
          L2: disk DEM cache (DEMSourceManager)
          L3: мережа (HTTP до OpenTopography/AWS)

        Args:
            tile_x, tile_y, tile_z: XYZ адреса тайлу
            source:        DEM джерело
            max_vertices:  ліміт вершин
            skirt_height_m: висота skirt

        Returns:
            dict сумісний з ResponseTilePayload (TerrainMesh.to_dict())
        """
        cache_key = f"{tile_z}/{tile_x}/{tile_y}:{source}:{max_vertices}"

        # L1 кеш
        if cache_key in self._mesh_cache:
            log.debug("terrain.mesh_cache.hit", key=cache_key)
            return self._mesh_cache[cache_key]

        log.info("terrain.mesh.build", tile=f"{tile_z}/{tile_x}/{tile_y}", source=source)

        # Виконуємо важку роботу в thread pool
        loop  = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _thread_pool,
            self._build_tile_mesh_sync,
            tile_x, tile_y, tile_z, source, max_vertices, skirt_height_m,
        )

        # Зберегти в L1 кеш (LRU eviction)
        if len(self._mesh_cache) >= self._max_cache:
            oldest = next(iter(self._mesh_cache))
            del self._mesh_cache[oldest]
        self._mesh_cache[cache_key] = result

        return result

    async def get_tile_meta(
        self,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        source: str = "copernicus25",
    ) -> dict:
        """Отримати метадані тайлу без mesh."""
        from geoengine.geo.projection import TileXYZ, tile_to_bbox

        tile = TileXYZ(x=tile_x, y=tile_y, z=tile_z)
        bbox = tile_to_bbox(tile)

        loop = asyncio.get_event_loop()
        dem_tile = await self._source_mgr.fetch(
            bbox=bbox,
            source=self._DEMSourceID(source),
        )

        return {
            "tile":           {"x": tile_x, "y": tile_y, "z": tile_z},
            "bbox":           bbox.to_list(),
            "min_elevation":  dem_tile.min_elevation,
            "max_elevation":  dem_tile.max_elevation,
            "mean_elevation": dem_tile.mean_elevation,
            "coverage_pct":   dem_tile.coverage_pct,
            "resolution_m":   dem_tile.resolution_x * 111_320,
            "source":         source,
        }

    async def get_elevations(
        self,
        points: list[tuple[float, float]],
        source: str = "copernicus25",
    ) -> list[float | None]:
        """
        Висоти для списку точок.
        Групує точки по тайлах для ефективності.
        """
        from geoengine.geo.projection import latlon_to_tile, tile_to_bbox

        # Групуємо точки по zoom=12 тайлах
        ZOOM = 12
        tile_groups: dict[str, list[tuple[int, float, float]]] = {}

        for i, (lat, lon) in enumerate(points):
            tile = latlon_to_tile(lat, lon, ZOOM)
            key  = f"{tile.z}/{tile.x}/{tile.y}"
            if key not in tile_groups:
                tile_groups[key] = []
            tile_groups[key].append((i, lat, lon))

        # Завантажуємо DEM для кожного унікального тайлу паралельно
        results: list[float | None] = [None] * len(points)

        tasks = []
        for key, group in tile_groups.items():
            z, x, y = map(int, key.split("/"))
            tasks.append(
                self._sample_tile_points(x, y, z, source, group, results)
            )

        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def get_bbox_mesh(
        self,
        bbox:         Any,  # BBox
        source:       str = "copernicus25",
        max_vertices: int = 65_536,
        lod_level:    int = 0,
    ) -> dict:
        """Mesh для довільного BBox."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _thread_pool,
            self._build_bbox_mesh_sync,
            bbox, source, max_vertices, lod_level,
        )

    async def get_tile_png(
        self,
        tile_x:   int,
        tile_y:   int,
        tile_z:   int,
        source:   str = "terrarium",
        colormap: str = "terrain",
    ) -> bytes:
        """Тайл як PNG зображення."""
        from geoengine.geo.projection import TileXYZ, tile_to_bbox

        tile = TileXYZ(x=tile_x, y=tile_y, z=tile_z)
        bbox = tile_to_bbox(tile)

        dem_tile = await self._source_mgr.fetch(
            bbox=bbox,
            source=self._DEMSourceID(source),
        )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _thread_pool,
            self._render_tile_png,
            dem_tile, colormap,
        )

    async def get_cache_stats(self) -> dict:
        """Статистика кешу."""
        disk_stats = self._source_mgr.cache_info()
        return {
            "files":   disk_stats["files"],
            "size_mb": disk_stats["size_mb"],
            "sources": {"mesh_memory": len(self._mesh_cache)},
        }

    async def clear_cache(self, source: str | None = None) -> int:
        """Очистити DEM кеш."""
        from geoengine.dem.sources import DEMSourceID
        src = DEMSourceID(source) if source else None
        self._mesh_cache.clear()
        return self._source_mgr.clear_cache(source=src)

    # ---- Sync методи (виконуються в ThreadPool) ----

    def _build_tile_mesh_sync(
        self,
        tile_x:        int,
        tile_y:        int,
        tile_z:        int,
        source:        str,
        max_vertices:  int,
        skirt_height_m: float,
    ) -> dict:
        """CPU-bound: завантажити DEM + побудувати mesh синхронно."""
        import asyncio
        from geoengine.geo.projection import TileXYZ, tile_to_bbox
        from geoengine.geo.coords import LLH
        from geoengine.dem.sources import DEMSourceID
        from geoengine.mesh.terrain import TerrainMeshBuilder

        tile = TileXYZ(x=tile_x, y=tile_y, z=tile_z)
        bbox = tile_to_bbox(tile)

        # Синхронний fetch через новий event loop у потоці
        dem_tile = asyncio.run(
            self._source_mgr.fetch(
                bbox=bbox,
                source=DEMSourceID(source),
            )
        )

        # Origin = центр тайлу
        center_lat, center_lon = bbox.center
        origin = LLH(lat=center_lat, lon=center_lon, alt=0.0)

        builder = TerrainMeshBuilder(
            origin=origin,
            skirt_height=skirt_height_m,
        )
        mesh = builder.build(
            tile=dem_tile,
            method="uniform",
            max_verts=max_vertices,
            lod_level=tile_z,
        )

        result = mesh.to_dict()
        result["min_elevation"]  = dem_tile.min_elevation
        result["max_elevation"]  = dem_tile.max_elevation
        result["resolution_m"]   = dem_tile.resolution_x * 111_320
        result["memory_bytes"]   = mesh.memory_bytes
        result["source"]         = source

        return result

    def _build_bbox_mesh_sync(
        self,
        bbox:         Any,
        source:       str,
        max_vertices: int,
        lod_level:    int,
    ) -> dict:
        """CPU-bound: mesh для bbox."""
        import asyncio
        from geoengine.dem.sources import DEMSourceID
        from geoengine.geo.coords import LLH
        from geoengine.mesh.terrain import TerrainMeshBuilder

        dem_tile = asyncio.run(
            self._source_mgr.fetch(
                bbox=bbox,
                source=DEMSourceID(source),
            )
        )

        center_lat, center_lon = bbox.center
        origin  = LLH(lat=center_lat, lon=center_lon, alt=0.0)
        builder = TerrainMeshBuilder(origin=origin)
        mesh    = builder.build(
            tile=dem_tile,
            max_verts=max_vertices,
            lod_level=lod_level,
        )

        result = mesh.to_dict()
        result["min_elevation"] = dem_tile.min_elevation
        result["max_elevation"] = dem_tile.max_elevation
        result["resolution_m"]  = dem_tile.resolution_x * 111_320
        result["memory_bytes"]  = mesh.memory_bytes
        result["source"]        = source
        return result

    def _render_tile_png(self, dem_tile: Any, colormap: str) -> bytes:
        """Рендерити DEM тайл як PNG bytes."""
        import matplotlib
        matplotlib.use("Agg")  # без GUI
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        data = dem_tile.data.copy()
        data[np.isnan(data)] = float(np.nanmin(data))  # nodata → мінімум

        # Нормалізація
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if vmax == vmin:
            vmax = vmin + 1

        cmap_map = {
            "terrain":   "terrain",
            "grayscale": "gray",
            "hillshade": "gray",
        }
        cmap_name = cmap_map.get(colormap, "terrain")
        cmap = cm.get_cmap(cmap_name)

        normalized = (data - vmin) / (vmax - vmin)
        rgba = cmap(normalized)
        rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)

        from PIL import Image
        img = Image.fromarray(rgb, "RGB").resize((256, 256))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    async def _sample_tile_points(
        self,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        source: str,
        group:  list[tuple[int, float, float]],
        results: list[float | None],
    ) -> None:
        """Вибірка висот для групи точок одного тайлу."""
        from geoengine.geo.projection import TileXYZ, tile_to_bbox

        tile = TileXYZ(x=tile_x, y=tile_y, z=tile_z)
        bbox = tile_to_bbox(tile)

        try:
            dem_tile = await self._source_mgr.fetch(
                bbox=bbox,
                source=self._DEMSourceID(source),
            )
            for idx, lat, lon in group:
                results[idx] = dem_tile.sample(lat, lon)
        except Exception as exc:
            log.warning(
                "terrain.sample.error",
                tile=f"{tile_z}/{tile_x}/{tile_y}",
                error=str(exc),
            )


# ---- Dependency Injection ----

@lru_cache(maxsize=1)
def get_terrain_service() -> TerrainService:
    """FastAPI Depends — singleton TerrainService."""
    return TerrainService()
