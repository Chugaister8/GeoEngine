"""
GeoEngine — Terrain Service
Центральний сервіс для DEM операцій на сервері.

Відповідає за:
  - Завантаження DEM тайлів (Terrarium, SRTM, Copernicus)
  - Побудову TerrainMesh для WebSocket стримінгу
  - L1 кеш (in-memory LRU) + L2 кеш (disk)
  - Async + ThreadPool для CPU-bound операцій
  - PNG preview тайлів для REST API

Lifecycle:
  TerrainService.create() → використання → TerrainService.close()
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from geoengine.dem.loader   import DEMLoader, DEMTile
from geoengine.dem.sources  import DEMSourceManager, DEMSourceID
from geoengine.dem.processor import merge_tiles, fill_gaps
from geoengine.geo.bbox     import BBox
from geoengine.geo.projection import tile_to_bbox, TileXYZ
from geoengine.geo.coords   import LLH
from geoengine.mesh.terrain import TerrainMeshBuilder
from geoengine.mesh.lod     import LODManager

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# LRU CACHE (простий in-memory)
# ----------------------------------------------------------------

class _LRUCache:
    """Простий thread-safe LRU кеш."""

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize  = maxsize
        self._cache:   dict[str, Any]   = {}
        self._access:  dict[str, float] = {}
        self._lock     = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key in self._cache:
                self._access[key] = time.monotonic()
                return self._cache[key]
        return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if len(self._cache) >= self._maxsize:
                # Видалити найстаріший елемент
                oldest = min(self._access, key=lambda k: self._access[k])
                del self._cache[oldest]
                del self._access[oldest]
            self._cache[key]  = value
            self._access[key] = time.monotonic()

    async def clear(self) -> int:
        async with self._lock:
            n = len(self._cache)
            self._cache.clear()
            self._access.clear()
        return n

    @property
    def size(self) -> int:
        return len(self._cache)


# ----------------------------------------------------------------
# TERRAIN SERVICE
# ----------------------------------------------------------------

class TerrainService:
    """
    Сервіс для роботи з DEM даними.

    Usage:
        svc = await TerrainService.create(config)
        mesh = await svc.get_tile_mesh(x=284, y=178, z=9)
        await svc.close()
    """

    def __init__(
        self,
        dem_cache_dir:   str | Path = "~/.geoengine/dem_cache",
        max_workers:     int        = 4,
        l1_cache_size:   int        = 128,
        api_keys:        dict[str, str] | None = None,
    ) -> None:
        self._cache_dir   = Path(dem_cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._executor    = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="terrain",
        )
        self._dem_manager = DEMSourceManager(
            cache_dir=self._cache_dir,
            api_keys=api_keys or {},
        )
        self._loader      = DEMLoader(fill_nodata=True)

        # L1 in-memory кеш (mesh dict — вже серіалізовані)
        self._mesh_cache  = _LRUCache(maxsize=l1_cache_size)
        # L1 кеш для DEM тайлів
        self._dem_cache   = _LRUCache(maxsize=l1_cache_size * 2)

        log.info(
            "terrain_service.init",
            cache_dir=str(self._cache_dir),
            workers=max_workers,
        )

    @classmethod
    async def create(
        cls,
        dem_cache_dir: str | Path = "~/.geoengine/dem_cache",
        max_workers:   int = 4,
        api_keys:      dict[str, str] | None = None,
    ) -> "TerrainService":
        """Фабричний метод з async ініціалізацією."""
        svc = cls(
            dem_cache_dir=dem_cache_dir,
            max_workers=max_workers,
            api_keys=api_keys,
        )
        return svc

    # ----------------------------------------------------------------
    # ПУБЛІЧНИЙ API
    # ----------------------------------------------------------------

    async def get_tile_mesh(
        self,
        x:              int,
        y:              int,
        z:              int,
        source:         str   = "terrarium",
        max_vertices:   int   = 65_536,
        skirt_height_m: float = 200.0,
        lod_level:      int   = 0,
    ) -> dict[str, Any]:
        """
        Отримати TerrainMesh для тайлу як wire-format dict.

        Порядок дій:
          1. L1 кеш (memory)
          2. Завантажити DEM тайл
          3. Побудувати mesh (CPU bound → ThreadPool)
          4. Серіалізувати → base64 буфери
          5. Зберегти в L1 кеш

        Returns:
            dict з vertex_count, triangle_count, buffers (base64),
            bbox, origin, min/max elevation

        Raises:
            ValueError: невірний тайл або джерело
        """
        cache_key = f"mesh:{z}/{x}/{y}:{source}:{max_vertices}"

        # L1 кеш
        cached = await self._mesh_cache.get(cache_key)
        if cached is not None:
            log.debug("terrain.mesh.cache_hit", tile=f"{z}/{x}/{y}")
            return cached

        # Завантажити DEM
        dem_tile = await self._get_dem_tile(x, y, z, source)

        # Побудувати mesh у ThreadPool
        mesh_data = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._build_mesh,
            dem_tile, max_vertices, skirt_height_m, lod_level,
        )

        # Зберегти у L1 кеш
        await self._mesh_cache.set(cache_key, mesh_data)
        return mesh_data

    async def get_tile_png(
        self,
        x:        int,
        y:        int,
        z:        int,
        source:   str = "terrarium",
        colormap: str = "terrain",
        size:     int = 256,
    ) -> bytes:
        """
        Отримати тайл як PNG зображення.

        Args:
            colormap: "terrain" | "hillshade" | "slope" | "gray"

        Returns:
            PNG bytes
        """
        cache_key = f"png:{z}/{x}/{y}:{source}:{colormap}"
        cached = await self._dem_cache.get(cache_key)
        if cached is not None:
            return cached

        dem_tile = await self._get_dem_tile(x, y, z, source)

        png_bytes = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._render_png,
            dem_tile, colormap, size,
        )

        await self._dem_cache.set(cache_key, png_bytes)
        return png_bytes

    async def get_tile_meta(
        self,
        x:      int,
        y:      int,
        z:      int,
        source: str = "terrarium",
    ) -> dict[str, Any]:
        """Метадані тайлу без побудови mesh."""
        dem_tile = await self._get_dem_tile(x, y, z, source)

        return {
            "tile":           {"x": x, "y": y, "z": z},
            "source":         source,
            "width":          dem_tile.width,
            "height":         dem_tile.height,
            "min_elevation":  round(float(dem_tile.min_elevation), 2),
            "max_elevation":  round(float(dem_tile.max_elevation), 2),
            "mean_elevation": round(float(dem_tile.mean_elevation), 2),
            "coverage_pct":   round(float(dem_tile.coverage_pct), 2),
            "resolution_m":   round(float(dem_tile.resolution_x * 111_320), 1),
            "bbox":           dem_tile.bbox.to_list(),
            "crs":            dem_tile.crs,
        }

    async def get_elevations(
        self,
        points: list[tuple[float, float]],
        source: str = "terrarium",
        zoom:   int = 11,
    ) -> list[float | None]:
        """
        Отримати висоти для списку точок (batch).

        Args:
            points: список (lat, lon)
            source: DEM джерело
            zoom:   zoom рівень для вибору тайлу

        Returns:
            Список висот у метрах (None якщо немає даних)
        """
        if not points:
            return []

        from geoengine.geo.projection import latlon_to_tile

        # Групуємо точки по тайлах
        tile_groups: dict[str, tuple[TileXYZ, list[int]]] = {}

        for i, (lat, lon) in enumerate(points):
            tile = latlon_to_tile(lat, lon, zoom)
            key  = f"{tile.z}/{tile.x}/{tile.y}"
            if key not in tile_groups:
                tile_groups[key] = (tile, [])
            tile_groups[key][1].append(i)

        # Завантажити кожен унікальний тайл
        results: list[float | None] = [None] * len(points)

        for key, (tile, indices) in tile_groups.items():
            try:
                dem_tile = await self._get_dem_tile(
                    tile.x, tile.y, tile.z, source
                )
                for i in indices:
                    lat, lon = points[i]
                    results[i] = dem_tile.sample(lat, lon)
            except Exception as exc:
                log.warning(
                    "terrain.elevation_error",
                    tile=key,
                    error=str(exc)[:80],
                )

        return results

    async def get_sources(self) -> list[dict[str, Any]]:
        """Список доступних DEM джерел."""
        return [
            {
                "id":          "terrarium",
                "name":        "AWS Terrarium",
                "resolution":  "~30m",
                "coverage":    "Global",
                "requires_key": False,
                "zoom_range":  [0, 15],
            },
            {
                "id":          "srtm30",
                "name":        "SRTM 30m",
                "resolution":  "30m",
                "coverage":    "56°S - 60°N",
                "requires_key": True,
                "zoom_range":  [0, 13],
            },
            {
                "id":          "copernicus25",
                "name":        "Copernicus DEM 25m",
                "resolution":  "25m",
                "coverage":    "Global (90m free)",
                "requires_key": True,
                "zoom_range":  [0, 14],
            },
        ]

    async def cache_stats(self) -> dict[str, Any]:
        """Статистика кешу."""
        disk_files = list(self._cache_dir.glob("**/*.png"))
        disk_size  = sum(f.stat().st_size for f in disk_files) / 1024 / 1024

        return {
            "l1_mesh_size":   self._mesh_cache.size,
            "l1_dem_size":    self._dem_cache.size,
            "files":          len(disk_files),
            "size_mb":        round(disk_size, 2),
            "cache_dir":      str(self._cache_dir),
        }

    async def clear_cache(self, source: str = "all") -> dict[str, Any]:
        """Очистити кеш."""
        mesh_cleared = await self._mesh_cache.clear()
        dem_cleared  = await self._dem_cache.clear()

        # Disk cache
        disk_cleared = 0
        if source == "all":
            for f in self._cache_dir.glob("**/*.png"):
                f.unlink()
                disk_cleared += 1
            for f in self._cache_dir.glob("**/*.tif"):
                f.unlink()
                disk_cleared += 1

        log.info(
            "terrain.cache.clear",
            mesh=mesh_cleared,
            dem=dem_cleared,
            disk=disk_cleared,
        )
        return {
            "l1_mesh_cleared":  mesh_cleared,
            "l1_dem_cleared":   dem_cleared,
            "disk_cleared":     disk_cleared,
            "source":           source,
        }

    async def close(self) -> None:
        """Graceful shutdown."""
        self._executor.shutdown(wait=False)
        log.info("terrain_service.closed")

    # ----------------------------------------------------------------
    # ПРИВАТНІ МЕТОДИ
    # ----------------------------------------------------------------

    async def _get_dem_tile(
        self,
        x:      int,
        y:      int,
        z:      int,
        source: str,
    ) -> DEMTile:
        """
        Завантажити DEM тайл.
        L1 cache → DEMSourceManager (disk + HTTP).
        """
        cache_key = f"dem:{z}/{x}/{y}:{source}"
        cached = await self._dem_cache.get(cache_key)
        if cached is not None:
            return cached

        # Конвертуємо tile → bbox
        tile = TileXYZ(x=x, y=y, z=z)
        bbox = tile_to_bbox(tile)

        # Завантажуємо через DEMSourceManager
        try:
            source_id = DEMSourceID(source)
        except ValueError:
            raise ValueError(
                f"Невідоме DEM джерело: {source!r}. "
                f"Доступні: {[s.value for s in DEMSourceID]}"
            )

        t0 = time.perf_counter()
        tiles = await self._dem_manager.fetch_tiles(
            bbox=bbox,
            zoom=min(z, 13),
            source=source_id,
        )

        if not tiles:
            raise ValueError(
                f"DEM дані недоступні для тайлу {z}/{x}/{y} "
                f"з джерела {source}"
            )

        # Merge якщо кілька тайлів
        dem_tile = tiles[0] if len(tiles) == 1 else merge_tiles(tiles)

        # Заповнити прогалини
        if dem_tile.has_nodata:
            dem_tile = fill_gaps(dem_tile, method="nearest")

        elapsed = (time.perf_counter() - t0) * 1000
        log.debug(
            "terrain.dem_loaded",
            tile=f"{z}/{x}/{y}",
            source=source,
            size=f"{dem_tile.width}×{dem_tile.height}",
            ms=round(elapsed, 1),
        )

        await self._dem_cache.set(cache_key, dem_tile)
        return dem_tile

    @staticmethod
    def _build_mesh(
        dem_tile:       DEMTile,
        max_vertices:   int,
        skirt_height_m: float,
        lod_level:      int,
    ) -> dict[str, Any]:
        """
        CPU-bound: побудувати TerrainMesh та серіалізувати.
        Виконується у ThreadPoolExecutor.
        """
        c      = dem_tile.bbox.center
        origin = LLH(lat=c[0], lon=c[1])

        builder = TerrainMeshBuilder(
            origin=origin,
            skirt_height=skirt_height_m,
        )
        mesh = builder.build(
            dem_tile,
            max_verts=max_vertices,
            lod_level=lod_level,
        )

        # Серіалізуємо у wire-format
        data = mesh.to_dict()

        # Додаємо origin та bbox до відповіді
        data["origin"] = {
            "lat": float(c[0]),
            "lon": float(c[1]),
            "alt": 0.0,
        }
        data["bbox"] = {
            "west":  dem_tile.bbox.west,
            "south": dem_tile.bbox.south,
            "east":  dem_tile.bbox.east,
            "north": dem_tile.bbox.north,
        }
        data["source"]         = "generated"
        data["min_elevation"]  = float(dem_tile.min_elevation)
        data["max_elevation"]  = float(dem_tile.max_elevation)
        data["lod_level"]      = lod_level

        return data

    @staticmethod
    def _render_png(
        dem_tile: DEMTile,
        colormap: str,
        size:     int,
    ) -> bytes:
        """
        CPU-bound: render DEM як PNG.
        Виконується у ThreadPoolExecutor.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from scipy.ndimage import zoom as scipy_zoom

        data = dem_tile.data.copy()

        # Resize до потрібного розміру
        h, w = data.shape
        if h != size or w != size:
            zoom_y = size / h
            zoom_x = size / w
            data = scipy_zoom(data, (zoom_y, zoom_x), order=1)

        # Замінюємо NaN
        valid = data[~np.isnan(data)]
        vmin  = float(valid.min()) if len(valid) else 0.0
        vmax  = float(valid.max()) if len(valid) else 1.0
        data  = np.where(np.isnan(data), vmin, data)

        fig, ax = plt.subplots(figsize=(size/96, size/96), dpi=96)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)

        cmap_map = {
            "terrain":   "terrain",
            "hillshade": "gray",
            "slope":     "RdYlGn_r",
            "gray":      "gray",
            "viridis":   "viridis",
        }
        cmap = cmap_map.get(colormap, "terrain")

        if colormap == "hillshade":
            from matplotlib.colors import LightSource
            ls    = LightSource(azdeg=315, altdeg=45)
            shade = ls.hillshade(data, vert_exag=2)
            ax.imshow(shade, cmap="gray", origin="upper",
                      interpolation="bilinear")
        else:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                      origin="upper", interpolation="bilinear")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0, dpi=96)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
