"""
GeoEngine — Terrain Mesh Generator
Конвертує DEMTile (heightmap) у 3D mesh для рендерингу.

Алгоритм:
  Heightmap → Vertices (XYZ) → Indices (трикутники) → UVs → Normals

Результат — TerrainMesh готовий для передачі у WebGPU рендерер
через WebSocket або прямий JS міст.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import structlog

from ..geo.bbox import BBox
from ..geo.coords import LLH, llh_to_enu
from ..dem.loader import DEMTile

log: structlog.BoundLogger = structlog.get_logger(__name__)


# ----------------------------------------------------------------
# ТИПИ РЕЗУЛЬТАТУ
# ----------------------------------------------------------------

@dataclass(slots=True)
class TerrainMesh:
    """
    3D меш терейну готовий для WebGPU рендерингу.

    Формат вершин сумісний з Three.js BufferGeometry:
      vertices:  Float32Array (N×3) — XYZ у метрах (ENU координати)
      indices:   Uint32Array  (M×3) — трикутники
      uvs:       Float32Array (N×2) — текстурні координати [0..1]
      normals:   Float32Array (N×3) — нормалі поверхні

    Координатна система:
      X = East  (метри від origin)
      Y = Up    (висота в метрах)
      Z = -North (Three.js: Z дивиться на глядача, тому -North)

    origin: LLH точка що є (0,0,0) у world-space
    """
    vertices:   npt.NDArray[np.float32]   # (N, 3)
    indices:    npt.NDArray[np.uint32]    # (M, 3)
    uvs:        npt.NDArray[np.float32]   # (N, 2)
    normals:    npt.NDArray[np.float32]   # (N, 3)
    bbox:       BBox
    origin:     LLH
    lod_level:  int = 0
    source_res: float = 0.0   # вихідна роздільна здатність DEM (м/піксель)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def memory_bytes(self) -> int:
        """Приблизний розмір у пам'яті (байти)."""
        return (
            self.vertices.nbytes
            + self.indices.nbytes
            + self.uvs.nbytes
            + self.normals.nbytes
        )

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        """
        Серіалізувати у dict для JSON/WebSocket передачі.
        Масиви кодуються як base64 для ефективності.
        """
        import base64

        def to_b64(arr: npt.NDArray) -> str:
            return base64.b64encode(arr.tobytes()).decode("ascii")

        return {
            "type":        "terrain_mesh",
            "lod_level":   self.lod_level,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "bbox": self.bbox.to_list(),
            "origin": {
                "lat": self.origin.lat,
                "lon": self.origin.lon,
                "alt": self.origin.alt,
            },
            "buffers": {
                "vertices": to_b64(self.vertices),   # Float32, (N,3)
                "indices":  to_b64(self.indices),    # Uint32,  (M,3)
                "uvs":      to_b64(self.uvs),        # Float32, (N,2)
                "normals":  to_b64(self.normals),    # Float32, (N,3)
            },
            "dtype": {
                "vertices": "float32",
                "indices":  "uint32",
                "uvs":      "float32",
                "normals":  "float32",
            },
            "shape": {
                "vertices": list(self.vertices.shape),
                "indices":  list(self.indices.shape),
                "uvs":      list(self.uvs.shape),
                "normals":  list(self.normals.shape),
            },
        }

    def __repr__(self) -> str:
        return (
            f"TerrainMesh("
            f"verts={self.vertex_count:,}, "
            f"tris={self.triangle_count:,}, "
            f"lod={self.lod_level}, "
            f"mem={self.memory_mb:.1f}MB)"
        )


# ----------------------------------------------------------------
# TERRAIN MESH BUILDER
# ----------------------------------------------------------------

class TerrainMeshBuilder:
    """
    Будує TerrainMesh з DEMTile.

    Підтримує:
    - Рівномірну сітку (uniform grid) — проста, швидка
    - Адаптивну сітку (adaptive) — менше трикутників там де рельєф плоский
    - Skirt — "спідниця" по краях тайлу щоб не було щілин між LOD

    Usage:
        builder = TerrainMeshBuilder(
            origin=LLH(lat=48.0, lon=23.0),
            skirt_height=100.0,
        )
        mesh = builder.build(dem_tile)
    """

    def __init__(
        self,
        origin:         LLH | None = None,
        skirt_height:   float = 100.0,
        y_up:           bool = True,
    ) -> None:
        """
        Args:
            origin:        (0,0,0) у world-space.
                           None = центр першого тайлу.
            skirt_height:  висота "спідниці" що звисає під край тайлу
                           для усунення щілин між LOD рівнями (метри).
            y_up:          True = Y вгору (Three.js конвенція)
                           False = Z вгору (Blender конвенція)
        """
        self._origin       = origin
        self._skirt_height = skirt_height
        self._y_up         = y_up

    # ---- Публічний API ----

    def build(
        self,
        tile:     DEMTile,
        method:   Literal["uniform", "adaptive"] = "uniform",
        max_verts: int = 65_536,
        lod_level: int = 0,
    ) -> TerrainMesh:
        """
        Збудувати меш з DEM тайлу.

        Args:
            tile:      вхідний DEM тайл
            method:    "uniform" — регулярна сітка
                       "adaptive" — менше вершин на плоских ділянках
            max_verts: максимальна кількість вершин
                       (впливає на downsampling якщо tile великий)
            lod_level: рівень деталізації (для метаданих)

        Returns:
            TerrainMesh готовий для рендерингу
        """
        # Визначити origin якщо не задано
        origin = self._origin
        if origin is None:
            center_lat, center_lon = tile.bbox.center
            origin = LLH(lat=center_lat, lon=center_lon, alt=0.0)

        # Downsampling якщо tile занадто великий
        data = self._maybe_downsample(tile.data, max_verts)
        h, w = data.shape

        log.debug(
            "mesh.build.start",
            method=method,
            size=f"{w}×{h}",
            lod=lod_level,
        )

        if method == "uniform":
            vertices, indices, uvs = self._build_uniform(
                data, tile.bbox, origin, w, h
            )
        elif method == "adaptive":
            vertices, indices, uvs = self._build_adaptive(
                data, tile.bbox, origin, w, h
            )
        else:
            raise NotImplementedError(f"Метод '{method}' не підтримується")

        # Додати skirt (спідниця по краях)
        if self._skirt_height > 0:
            vertices, indices, uvs = self._add_skirt(
                vertices, indices, uvs, data, tile.bbox, origin, w, h
            )

        # Обчислити нормалі
        normals = _compute_normals(vertices, indices)

        mesh = TerrainMesh(
            vertices=vertices,
            indices=indices,
            uvs=uvs,
            normals=normals,
            bbox=tile.bbox,
            origin=origin,
            lod_level=lod_level,
            source_res=tile.resolution_x * 111_320.0,
        )

        log.info(
            "mesh.build.done",
            verts=f"{mesh.vertex_count:,}",
            tris=f"{mesh.triangle_count:,}",
            mem=f"{mesh.memory_mb:.1f}MB",
            lod=lod_level,
        )

        return mesh

    # ---- Uniform Grid ----

    def _build_uniform(
        self,
        data:   npt.NDArray[np.float32],
        bbox:   BBox,
        origin: LLH,
        w:      int,
        h:      int,
    ) -> tuple[
        npt.NDArray[np.float32],  # vertices (N,3)
        npt.NDArray[np.uint32],   # indices  (M,3)
        npt.NDArray[np.float32],  # uvs      (N,2)
    ]:
        """
        Регулярна сітка: N = w×h вершин, M = (w-1)×(h-1)×2 трикутники.

        Координати:
          lat лінійно від bbox.north (row=0) до bbox.south (row=h-1)
          lon лінійно від bbox.west  (col=0) до bbox.east  (col=w-1)

        ENU конвертація:
          East  = X
          Up    = Y  (якщо y_up=True)
          -North = Z (Three.js: Z дивиться на камеру)
        """
        N = w * h

        # Координатні сітки
        lons = np.linspace(bbox.west,  bbox.east,  w, dtype=np.float64)
        lats = np.linspace(bbox.north, bbox.south, h, dtype=np.float64)
        lon_grid, lat_grid = np.meshgrid(lons, lats)  # обидва (h, w)

        # Висоти (NaN → 0.0 для меша, але зберігаємо маску)
        elev = np.where(np.isnan(data), 0.0, data).astype(np.float64)

        # ENU конвертація (векторизована)
        enu = _latlon_grid_to_enu(lat_grid, lon_grid, elev, origin)
        # enu: (h, w, 3) — [east, north, up]

        # Перепакувати у Three.js конвенцію: [X=East, Y=Up, Z=-North]
        x = enu[:, :, 0]           # East
        y = enu[:, :, 2]           # Up (висота)
        z = -enu[:, :, 1]          # -North

        vertices = np.stack([x, y, z], axis=-1).reshape(N, 3).astype(np.float32)

        # UV координати [0..1]
        u = np.linspace(0.0, 1.0, w, dtype=np.float32)
        v = np.linspace(0.0, 1.0, h, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)
        uvs = np.stack([u_grid, v_grid], axis=-1).reshape(N, 2)

        # Індекси (два трикутники на квадрат)
        indices = _build_grid_indices(w, h)

        return vertices, indices, uvs

    # ---- Adaptive Grid ----

    def _build_adaptive(
        self,
        data:   npt.NDArray[np.float32],
        bbox:   BBox,
        origin: LLH,
        w:      int,
        h:      int,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.uint32],
        npt.NDArray[np.float32],
    ]:
        """
        Адаптивна сітка: більше трикутників там де висота змінюється,
        менше — де рельєф плоский.

        Алгоритм: обчислюємо локальну варіацію висот,
        потім downsampling flat регіонів через quadtree subdivision.

        Для спрощення реалізації — variance-based decimation:
        якщо варіація висот у квадраті < threshold → один quad.
        """
        # Variance map — де рельєф складний
        variance = _compute_local_variance(data, kernel=4)
        threshold = float(np.percentile(variance[~np.isnan(variance)], 30))

        # Adaptive grid як список quad patches
        quads = _adaptive_subdivide(data, variance, threshold, 0, 0, w - 1, h - 1)

        # Збираємо вершини з quad списку
        all_verts: list[npt.NDArray[np.float32]] = []
        all_uvs:   list[npt.NDArray[np.float32]] = []
        all_idx:   list[npt.NDArray[np.uint32]]  = []

        vert_offset = 0
        lons = np.linspace(bbox.west,  bbox.east,  w)
        lats = np.linspace(bbox.north, bbox.south, h)

        for (r0, c0, r1, c1) in quads:
            # Чотири кути quad
            corners_lat = np.array([lats[r0], lats[r0], lats[r1], lats[r1]])
            corners_lon = np.array([lons[c0], lons[c1], lons[c0], lons[c1]])
            corners_alt = np.array([
                float(data[r0, c0]) if not np.isnan(data[r0, c0]) else 0.0,
                float(data[r0, c1]) if not np.isnan(data[r0, c1]) else 0.0,
                float(data[r1, c0]) if not np.isnan(data[r1, c0]) else 0.0,
                float(data[r1, c1]) if not np.isnan(data[r1, c1]) else 0.0,
            ])

            enu_corners = np.array([
                _latlon_to_enu_single(corners_lat[i], corners_lon[i], corners_alt[i], origin)
                for i in range(4)
            ])

            verts = np.array([
                [enu_corners[i, 0], enu_corners[i, 2], -enu_corners[i, 1]]
                for i in range(4)
            ], dtype=np.float32)

            u_vals = np.array([c0/(w-1), c1/(w-1), c0/(w-1), c1/(w-1)], dtype=np.float32)
            v_vals = np.array([r0/(h-1), r0/(h-1), r1/(h-1), r1/(h-1)], dtype=np.float32)
            uvs_quad = np.stack([u_vals, v_vals], axis=-1)

            # Два трикутники: [0,1,2] та [1,3,2]
            idx = np.array([
                [0, 1, 2],
                [1, 3, 2],
            ], dtype=np.uint32) + vert_offset

            all_verts.append(verts)
            all_uvs.append(uvs_quad)
            all_idx.append(idx)
            vert_offset += 4

        if not all_verts:
            # fallback до uniform
            return self._build_uniform(data, bbox, origin, w, h)

        vertices = np.concatenate(all_verts, axis=0)
        uvs      = np.concatenate(all_uvs,   axis=0)
        indices  = np.concatenate(all_idx,   axis=0)

        return vertices, indices, uvs

    # ---- Skirt ----

    def _add_skirt(
        self,
        vertices: npt.NDArray[np.float32],
        indices:  npt.NDArray[np.uint32],
        uvs:      npt.NDArray[np.float32],
        data:     npt.NDArray[np.float32],
        bbox:     BBox,
        origin:   LLH,
        w:        int,
        h:        int,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.uint32],
        npt.NDArray[np.float32],
    ]:
        """
        Додати "спідницю" — вертикальні стіни по краях тайлу
        що звисають вниз на skirt_height метрів.

        Усуває щілини між тайлами різних LOD рівнів.

        Порядок країв: top (row=0), bottom (row=h-1), left (col=0), right (col=w-1)
        """
        skirt_drop = self._skirt_height
        new_verts: list[npt.NDArray[np.float32]] = [vertices]
        new_uvs:   list[npt.NDArray[np.float32]] = [uvs]
        new_idx:   list[npt.NDArray[np.uint32]]  = [indices]
        base_offset = len(vertices)

        # Для кожного краю — беремо крайні вершини та опускаємо вниз
        # Uniform grid: вершини впорядковані row-major (row * w + col)
        edges = [
            [row * w + 0       for row in range(h)],   # left edge
            [row * w + (w - 1) for row in range(h)],   # right edge
            [0  * w + col      for col in range(w)],   # top edge
            [(h-1) * w + col   for col in range(w)],   # bottom edge
        ]

        for edge_verts_idx in edges:
            n = len(edge_verts_idx)
            if n < 2:
                continue

            # Беремо позиції крайніх вершин
            edge_positions = vertices[edge_verts_idx]   # (n, 3)
            edge_uvs       = uvs[edge_verts_idx]        # (n, 2)

            # Опускаємо Y (висота) на skirt_drop
            skirt_positions        = edge_positions.copy()
            skirt_positions[:, 1] -= skirt_drop

            new_verts.append(skirt_positions)
            new_uvs.append(edge_uvs)

            # Трикутники між краєм та спідницею
            skirt_offset = base_offset
            skirt_idx: list[list[int]] = []
            for i in range(n - 1):
                top0 = edge_verts_idx[i]
                top1 = edge_verts_idx[i + 1]
                bot0 = skirt_offset + i
                bot1 = skirt_offset + i + 1
                skirt_idx.extend([
                    [top0, top1, bot0],
                    [top1, bot1, bot0],
                ])

            new_idx.append(np.array(skirt_idx, dtype=np.uint32))
            base_offset += n

        return (
            np.concatenate(new_verts, axis=0),
            np.concatenate(new_idx,   axis=0),
            np.concatenate(new_uvs,   axis=0),
        )

    # ---- Утиліти ----

    @staticmethod
    def _maybe_downsample(
        data:      npt.NDArray[np.float32],
        max_verts: int,
    ) -> npt.NDArray[np.float32]:
        """
        Downsample якщо кількість вершин перевищує max_verts.
        Використовує box-averaging (не bilinear) для збереження рельєфу.
        """
        h, w = data.shape
        if h * w <= max_verts:
            return data

        # Цільовий розмір зберігаючи пропорції
        ratio = (max_verts / (h * w)) ** 0.5
        new_h = max(2, int(h * ratio))
        new_w = max(2, int(w * ratio))

        from scipy.ndimage import zoom as scipy_zoom
        scale = (new_h / h, new_w / w)
        return scipy_zoom(
            np.where(np.isnan(data), 0.0, data),
            scale, order=1
        ).astype(np.float32)


# ----------------------------------------------------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# ----------------------------------------------------------------

def _build_grid_indices(w: int, h: int) -> npt.NDArray[np.uint32]:
    """
    Побудувати індекси трикутників для регулярної w×h сітки.

    Кожен квадрат → 2 трикутники:
      [i*w+j,   i*w+j+1, (i+1)*w+j  ]
      [i*w+j+1, (i+1)*w+j+1, (i+1)*w+j]

    Порядок counter-clockwise (CCW) — стандарт WebGPU/Three.js.
    """
    rows = h - 1
    cols = w - 1
    n_quads = rows * cols

    # Vectorized побудова
    row_idx = np.arange(rows, dtype=np.uint32).repeat(cols)
    col_idx = np.tile(np.arange(cols, dtype=np.uint32), rows)

    tl = row_idx * w + col_idx          # top-left
    tr = row_idx * w + col_idx + 1      # top-right
    bl = (row_idx + 1) * w + col_idx    # bottom-left
    br = (row_idx + 1) * w + col_idx + 1  # bottom-right

    # Два трикутники на квадрат: CCW
    tri1 = np.stack([tl, bl, tr], axis=-1)  # (n_quads, 3)
    tri2 = np.stack([tr, bl, br], axis=-1)  # (n_quads, 3)

    return np.concatenate([tri1, tri2], axis=0).astype(np.uint32)


def _latlon_grid_to_enu(
    lat_grid: npt.NDArray[np.float64],
    lon_grid: npt.NDArray[np.float64],
    alt_grid: npt.NDArray[np.float64],
    origin:   LLH,
) -> npt.NDArray[np.float64]:
    """
    Векторизована конвертація lat/lon/alt сітки → ENU (метри).
    Результат: (h, w, 3) — [east, north, up]

    Для тайлів до ~500км ×500км похибка < 1м (плоска Земля апроксимація).
    Для більших областей треба camera-relative rendering.
    """
    import math

    lat_r = np.radians(lat_grid)
    lon_r = np.radians(lon_grid)
    origin_lat_r = math.radians(origin.lat)
    origin_lon_r = math.radians(origin.lon)

    # Спрощена ENU (достатньо для тайлів до 500км)
    # East:  delta_lon * cos(lat) * R
    # North: delta_lat * R
    # Up:    delta_alt
    R = 6_378_137.0

    east  = (lon_r - origin_lon_r) * np.cos(origin_lat_r) * R
    north = (lat_r - origin_lat_r) * R
    up    = alt_grid - origin.alt

    return np.stack([east, north, up], axis=-1)


def _latlon_to_enu_single(
    lat:    float,
    lon:    float,
    alt:    float,
    origin: LLH,
) -> tuple[float, float, float]:
    """Конвертувати одну точку lat/lon/alt → ENU."""
    import math
    R = 6_378_137.0
    east  = (math.radians(lon) - math.radians(origin.lon)) * math.cos(math.radians(origin.lat)) * R
    north = (math.radians(lat) - math.radians(origin.lat)) * R
    up    = alt - origin.alt
    return east, north, up


def _compute_normals(
    vertices: npt.NDArray[np.float32],
    indices:  npt.NDArray[np.uint32],
) -> npt.NDArray[np.float32]:
    """
    Обчислити нормалі вершин через усереднення нормалей прилеглих трикутників.

    Алгоритм:
    1. Для кожного трикутника — cross product двох ребер → normal
    2. Для кожної вершини — сума нормалей прилеглих трикутників
    3. Нормалізація до одиничного вектора
    """
    normals = np.zeros_like(vertices)  # (N, 3)

    # Вершини трикутників
    v0 = vertices[indices[:, 0]]  # (M, 3)
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    # Нормалі трикутників (cross product)
    edge1 = v1 - v0
    edge2 = v2 - v0
    tri_normals = np.cross(edge1, edge2)  # (M, 3)

    # Накопичуємо нормалі у вершинах
    np.add.at(normals, indices[:, 0], tri_normals)
    np.add.at(normals, indices[:, 1], tri_normals)
    np.add.at(normals, indices[:, 2], tri_normals)

    # Нормалізація
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)  # уникаємо ділення на 0
    normals = normals / lengths

    return normals.astype(np.float32)


def _compute_local_variance(
    data:   npt.NDArray[np.float32],
    kernel: int = 4,
) -> npt.NDArray[np.float32]:
    """
    Локальна варіація висот у вікні kernel×kernel.
    Використовується для adaptive mesh subdivision.
    """
    from scipy.ndimage import uniform_filter

    d = np.where(np.isnan(data), 0.0, data).astype(np.float64)
    mean  = uniform_filter(d,    size=kernel)
    mean2 = uniform_filter(d**2, size=kernel)
    variance = mean2 - mean**2
    return np.maximum(0, variance).astype(np.float32)


def _adaptive_subdivide(
    data:      npt.NDArray[np.float32],
    variance:  npt.NDArray[np.float32],
    threshold: float,
    r0: int, c0: int,
    r1: int, c1: int,
    min_size:  int = 4,
    max_depth: int = 6,
    depth:     int = 0,
) -> list[tuple[int, int, int, int]]:
    """
    Рекурсивний quadtree subdivision.
    Повертає список (r0, c0, r1, c1) quad патчів.
    """
    if depth >= max_depth:
        return [(r0, c0, r1, c1)]

    h = r1 - r0
    w = c1 - c0
    if h <= min_size or w <= min_size:
        return [(r0, c0, r1, c1)]

    # Перевірити варіацію у цьому квадраті
    region_var = variance[r0:r1, c0:c1]
    max_var    = float(np.nanmax(region_var)) if region_var.size > 0 else 0.0

    if max_var <= threshold:
        # Плоска ділянка — один quad
        return [(r0, c0, r1, c1)]

    # Ділимо на 4 квадранти
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    result: list[tuple[int, int, int, int]] = []
    for (sr0, sc0, sr1, sc1) in [
        (r0, c0, rm, cm),
        (r0, cm, rm, c1),
        (rm, c0, r1, cm),
        (rm, cm, r1, c1),
    ]:
        result.extend(_adaptive_subdivide(
            data, variance, threshold,
            sr0, sc0, sr1, sc1,
            min_size, max_depth, depth + 1,
        ))
    return result
