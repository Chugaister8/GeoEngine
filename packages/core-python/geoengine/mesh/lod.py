"""
GeoEngine — LOD System (Level of Detail)
Управління деталізацією терейну залежно від відстані до камери.

Архітектура:
  LODManager тримає піраміду мешів різних роздільних здатностей.
  При запиті — повертає відповідний LOD рівень.

  LOD 0: 1× resolution  (найближче,  найбільше трикутників)
  LOD 1: 4× downsampled
  LOD 2: 16× downsampled
  ...

Схема CDLOD (Continuous Distance LOD):
  Відстань від камери → плавний перехід між рівнями (geomorphing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import structlog

from ..geo.bbox import BBox
from ..geo.coords import LLH
from ..dem.loader import DEMTile
from .terrain import TerrainMesh, TerrainMeshBuilder

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# КОНФІГУРАЦІЯ LOD РІВНІВ
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LODConfig:
    """Конфігурація одного LOD рівня."""
    level:           int
    max_distance_m:  float     # максимальна відстань камери (м)
    resolution_m:    float     # роздільна здатність (м/піксель)
    max_vertices:    int       # максимальна кількість вершин тайлу


# Стандартна піраміда LOD для терейну
DEFAULT_LOD_PYRAMID: Final[list[LODConfig]] = [
    LODConfig(level=0, max_distance_m=2_000,    resolution_m=1,    max_vertices=65_536),
    LODConfig(level=1, max_distance_m=10_000,   resolution_m=5,    max_vertices=16_384),
    LODConfig(level=2, max_distance_m=50_000,   resolution_m=25,   max_vertices=4_096),
    LODConfig(level=3, max_distance_m=200_000,  resolution_m=100,  max_vertices=1_024),
    LODConfig(level=4, max_distance_m=1_000_000,resolution_m=500,  max_vertices=256),
    LODConfig(level=5, max_distance_m=float("inf"), resolution_m=2500, max_vertices=64),
]


# ----------------------------------------------------------------
# LOD MANAGER
# ----------------------------------------------------------------

class LODManager:
    """
    Менеджер LOD піраміди для одного тайлу.

    Зберігає кешовані меші для кожного LOD рівня,
    будує їх ліниво (тільки при першому запиті).

    Usage:
        manager = LODManager(dem_tile, origin=LLH(48, 23))
        mesh = manager.get_mesh_for_distance(camera_distance_m=5000)
    """

    def __init__(
        self,
        tile:     DEMTile,
        origin:   LLH | None = None,
        pyramid:  list[LODConfig] | None = None,
    ) -> None:
        self._tile    = tile
        self._origin  = origin
        self._pyramid = pyramid or DEFAULT_LOD_PYRAMID
        self._cache:  dict[int, TerrainMesh] = {}
        self._builder = TerrainMeshBuilder(
            origin=origin,
            skirt_height=200.0,
        )

    def get_mesh_for_distance(self, camera_distance_m: float) -> TerrainMesh:
        """
        Отримати меш відповідного LOD рівня для заданої відстані.

        Args:
            camera_distance_m: відстань від камери до центру тайлу (метри)

        Returns:
            TerrainMesh оптимального LOD рівня
        """
        level = self._select_lod_level(camera_distance_m)
        return self.get_mesh_for_level(level)

    def get_mesh_for_level(self, level: int) -> TerrainMesh:
        """
        Отримати меш конкретного LOD рівня.
        Будує ліниво і кешує.

        Args:
            level: LOD рівень (0 = найдетальніший)

        Returns:
            TerrainMesh для цього рівня
        """
        level = max(0, min(level, len(self._pyramid) - 1))

        if level not in self._cache:
            self._cache[level] = self._build_lod(level)

        return self._cache[level]

    def preload_all(self) -> None:
        """Попередньо побудувати всі LOD рівні."""
        for i in range(len(self._pyramid)):
            self.get_mesh_for_level(i)
        log.info("lod.preload.done", levels=len(self._pyramid))

    def clear_cache(self) -> None:
        """Очистити кеш мешів (звільнити пам'ять)."""
        self._cache.clear()

    @property
    def memory_usage_mb(self) -> float:
        """Загальне використання пам'яті всіма кешованими мешами."""
        return sum(m.memory_mb for m in self._cache.values())

    # ---- Приватні методи ----

    def _select_lod_level(self, distance_m: float) -> int:
        """Вибрати оптимальний LOD рівень для відстані."""
        for config in self._pyramid:
            if distance_m <= config.max_distance_m:
                return config.level
        return self._pyramid[-1].level

    def _build_lod(self, level: int) -> TerrainMesh:
        """Збудувати меш для конкретного LOD рівня."""
        config = self._pyramid[level]
        log.debug("lod.build", level=level, max_verts=config.max_vertices)

        return self._builder.build(
            tile=self._tile,
            method="uniform",
            max_verts=config.max_vertices,
            lod_level=level,
        )


# ----------------------------------------------------------------
# LOD TILE GRID — менеджер сітки тайлів
# ----------------------------------------------------------------

@dataclass(slots=True)
class TileGridCell:
    """Одна клітинка сітки тайлів з LOD менеджером."""
    tile_key:    str              # унікальний ключ (z/x/y)
    bbox:        BBox
    lod_manager: LODManager
    last_used:   float = 0.0     # Unix timestamp


class LODTileGrid:
    """
    Сітка тайлів з LOD управлінням.

    Тримає N×N активних тайлів навколо камери.
    Автоматично завантажує нові та вивантажує далекі (LRU).

    Usage:
        grid = LODTileGrid(origin=LLH(48, 23), max_tiles=64)
        grid.update(camera_pos=LLH(48.1, 23.2, 1000))
        meshes = grid.get_visible_meshes()
    """

    def __init__(
        self,
        origin:    LLH,
        max_tiles: int = 64,
        pyramid:   list[LODConfig] | None = None,
    ) -> None:
        self._origin   = origin
        self._max_tiles = max_tiles
        self._pyramid  = pyramid or DEFAULT_LOD_PYRAMID
        self._cells:   dict[str, TileGridCell] = {}

    def register_tile(self, key: str, tile: DEMTile) -> None:
        """Зареєструвати DEM тайл у сітці."""
        if key in self._cells:
            return

        if len(self._cells) >= self._max_tiles:
            self._evict_lru()

        manager = LODManager(
            tile=tile,
            origin=self._origin,
            pyramid=self._pyramid,
        )
        self._cells[key] = TileGridCell(
            tile_key=key,
            bbox=tile.bbox,
            lod_manager=manager,
        )
        log.debug("lod_grid.register", key=key, total=len(self._cells))

    def get_mesh(
        self,
        key:               str,
        camera_distance_m: float,
    ) -> TerrainMesh | None:
        """Отримати меш тайлу для заданої відстані камери."""
        import time
        cell = self._cells.get(key)
        if cell is None:
            return None
        cell.last_used = time.monotonic()
        return cell.lod_manager.get_mesh_for_distance(camera_distance_m)

    def get_all_visible(
        self,
        camera_pos: LLH,
        max_distance_m: float = 50_000,
    ) -> list[tuple[str, TerrainMesh]]:
        """
        Отримати всі видимі меші відсортовані від ближнього до дальнього.

        Args:
            camera_pos:     позиція камери
            max_distance_m: максимальна відстань видимості

        Returns:
            Список (tile_key, mesh) відсортованих за відстанню
        """
        from ..geo.coords import haversine_distance

        result: list[tuple[str, TerrainMesh, float]] = []

        for key, cell in self._cells.items():
            center_lat, center_lon = cell.bbox.center
            dist = haversine_distance(
                camera_pos,
                LLH(lat=center_lat, lon=center_lon),
            )
            if dist > max_distance_m:
                continue

            mesh = self.get_mesh(key, dist)
            if mesh is not None:
                result.append((key, mesh, dist))

        result.sort(key=lambda x: x[2])
        return [(key, mesh) for key, mesh, _ in result]

    def memory_usage_mb(self) -> float:
        """Загальне використання пам'яті."""
        return sum(
            cell.lod_manager.memory_usage_mb
            for cell in self._cells.values()
        )

    def _evict_lru(self) -> None:
        """Видалити найстаріший (least recently used) тайл."""
        if not self._cells:
            return
        lru_key = min(self._cells, key=lambda k: self._cells[k].last_used)
        del self._cells[lru_key]
        log.debug("lod_grid.evict", key=lru_key, remaining=len(self._cells))
