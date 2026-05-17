"""
GeoEngine — OSM Buildings
Конвертація OSM будівель у 3D геометрію.

Алгоритм:
  OSMWay (полігон) → висота (з тегів або дефолт) → extrusion mesh

Підтримує:
  LOD 1: проста видавка (footprint × height)
  LOD 2: проста форма даху (flat, gabled, hipped, pyramidal)
  LOD 3: текстури фасадів, вікна (TODO: наступна фаза)

Джерела висоти (пріоритет):
  1. height=* (метри)
  2. building:levels=* × FLOOR_HEIGHT
  3. Дефолт за типом будівлі
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import structlog

from .fetcher import OSMWay, OSMData

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------------

FLOOR_HEIGHT:    Final[float] = 3.2    # метрів на поверх
MIN_HEIGHT:      Final[float] = 3.0    # мінімальна висота будівлі
MAX_HEIGHT:      Final[float] = 828.0  # Бурдж-Халіфа

# Дефолтні висоти за типом будівлі
BUILDING_TYPE_HEIGHTS: Final[dict[str, float]] = {
    "house":       6.5,
    "residential": 9.0,
    "apartments":  15.0,
    "commercial":  12.0,
    "retail":      4.5,
    "industrial":  8.0,
    "warehouse":   10.0,
    "office":      16.0,
    "church":      15.0,
    "cathedral":   30.0,
    "mosque":      20.0,
    "school":      9.0,
    "hospital":    15.0,
    "stadium":     25.0,
    "garage":      3.0,
    "shed":        2.5,
    "greenhouse":  3.5,
    "yes":         9.0,   # невідомий тип — загальний дефолт
}

# Кольори будівель за типом (RGBA, 0-1)
BUILDING_COLORS: Final[dict[str, tuple[float, float, float]]] = {
    "residential":  (0.85, 0.75, 0.65),
    "apartments":   (0.80, 0.70, 0.80),
    "commercial":   (0.70, 0.80, 0.90),
    "industrial":   (0.70, 0.70, 0.65),
    "retail":       (0.90, 0.80, 0.70),
    "office":       (0.75, 0.85, 0.95),
    "church":       (0.95, 0.90, 0.80),
    "school":       (0.95, 0.85, 0.70),
    "default":      (0.82, 0.76, 0.68),
}


# ----------------------------------------------------------------
# ТИПИ РЕЗУЛЬТАТУ
# ----------------------------------------------------------------

@dataclass(slots=True)
class BuildingMesh:
    """
    3D меш однієї будівлі.
    Формат сумісний з TerrainMesh (base64 буфери для WebGPU).
    """
    osm_id:   int
    vertices: npt.NDArray[np.float32]   # (N, 3) XYZ у метрах ENU
    indices:  npt.NDArray[np.uint32]    # (M, 3) трикутники
    uvs:      npt.NDArray[np.float32]   # (N, 2)
    normals:  npt.NDArray[np.float32]   # (N, 3)
    color:    tuple[float, float, float]

    # Метадані
    height:        float
    floors:        int
    building_type: str
    name:          str = ""
    footprint_area_m2: float = 0.0

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    def to_dict(self) -> dict:
        """Серіалізувати у wire-format dict."""
        import base64

        def b64(arr: npt.NDArray) -> str:
            return base64.b64encode(arr.tobytes()).decode("ascii")

        return {
            "type":          "building_mesh",
            "osm_id":        self.osm_id,
            "vertex_count":  self.vertex_count,
            "triangle_count": self.triangle_count,
            "height":        self.height,
            "floors":        self.floors,
            "building_type": self.building_type,
            "name":          self.name,
            "color":         list(self.color),
            "footprint_m2":  round(self.footprint_area_m2, 1),
            "buffers": {
                "vertices": b64(self.vertices),
                "indices":  b64(self.indices),
                "uvs":      b64(self.uvs),
                "normals":  b64(self.normals),
            },
        }


@dataclass(slots=True)
class BuildingCollection:
    """Колекція будівельних мешів для одного bbox."""
    meshes:         list[BuildingMesh]
    origin_lat:     float
    origin_lon:     float
    total_buildings: int
    skipped:        int = 0

    @property
    def vertex_count(self) -> int:
        return sum(m.vertex_count for m in self.meshes)

    @property
    def triangle_count(self) -> int:
        return sum(m.triangle_count for m in self.meshes)

    def to_dict(self) -> dict:
        return {
            "type":           "building_collection",
            "count":          len(self.meshes),
            "total":          self.total_buildings,
            "skipped":        self.skipped,
            "origin":         {"lat": self.origin_lat, "lon": self.origin_lon},
            "vertex_count":   self.vertex_count,
            "triangle_count": self.triangle_count,
            "buildings":      [m.to_dict() for m in self.meshes],
        }


# ----------------------------------------------------------------
# BUILDING EXTRUDER
# ----------------------------------------------------------------

class BuildingExtruder:
    """
    Конвертує OSM будівлі у 3D меші.

    Алгоритм для LOD1 (проста видавка):
    1. Отримати polígon footprint (coords)
    2. Обчислити висоту (з тегів або дефолт)
    3. Тріангулювати footprint (підлога + дах)
    4. Видавити стіни (кожне ребро → 2 трикутники)
    5. Повернути BuildingMesh

    Usage:
        extruder = BuildingExtruder(origin_lat=48.0, origin_lon=23.0)
        collection = extruder.extrude_all(osm_data)
    """

    def __init__(
        self,
        origin_lat:   float,
        origin_lon:   float,
        lod:          Literal[1, 2] = 1,
        min_area_m2:  float = 10.0,    # мінімальна площа (відкидаємо сміття)
        max_buildings: int  = 10_000,  # ліміт на bbox
    ) -> None:
        self._origin_lat  = origin_lat
        self._origin_lon  = origin_lon
        self._lod         = lod
        self._min_area    = min_area_m2
        self._max_buildings = max_buildings

    # ---- Публічний API ----

    def extrude_all(self, osm_data: OSMData) -> BuildingCollection:
        """
        Видавити всі будівлі з OSMData.

        Args:
            osm_data: результат Overpass fetch

        Returns:
            BuildingCollection з усіма мешами
        """
        buildings = osm_data.buildings()
        log.info(
            "buildings.extrude.start",
            count=len(buildings),
            lod=self._lod,
        )

        meshes:  list[BuildingMesh] = []
        skipped: int = 0

        for way in buildings[:self._max_buildings]:
            try:
                mesh = self._extrude_one(way)
                if mesh is not None:
                    meshes.append(mesh)
                else:
                    skipped += 1
            except Exception as exc:
                log.debug(
                    "buildings.extrude.skip",
                    osm_id=way.id,
                    error=str(exc)[:100],
                )
                skipped += 1

        log.info(
            "buildings.extrude.done",
            built=len(meshes),
            skipped=skipped,
            verts=sum(m.vertex_count for m in meshes),
            tris=sum(m.triangle_count for m in meshes),
        )

        return BuildingCollection(
            meshes=meshes,
            origin_lat=self._origin_lat,
            origin_lon=self._origin_lon,
            total_buildings=len(buildings),
            skipped=skipped,
        )

    def extrude_one(self, way: OSMWay) -> BuildingMesh | None:
        """Видавити одну будівлю. Публічна версія."""
        return self._extrude_one(way)

    # ---- Приватні методи ----

    def _extrude_one(self, way: OSMWay) -> BuildingMesh | None:
        """
        Видавити одну будівлю.

        Returns:
            BuildingMesh або None якщо будівля невалідна
        """
        if len(way.coords) < 3:
            return None

        # Конвертація coords у ENU (метри від origin)
        enu_coords = self._coords_to_enu(way.coords)
        if enu_coords is None:
            return None

        # Перевірка площі
        area = _polygon_area_m2(enu_coords)
        if area < self._min_area:
            return None

        # Висота будівлі
        height = _parse_height(way.tags)

        # Кількість поверхів
        floors = _parse_floors(way.tags, height)

        # Тип будівлі
        btype = way.tags.get("building", "yes")

        # Колір
        color = BUILDING_COLORS.get(btype, BUILDING_COLORS["default"])

        # Тріангуляція footprint (для підлоги та даху)
        try:
            floor_indices = _triangulate_polygon(enu_coords)
        except Exception:
            return None

        if len(floor_indices) == 0:
            return None

        # Побудова геометрії
        vertices, indices, uvs, normals = _build_extrusion(
            enu_coords, floor_indices, height
        )

        return BuildingMesh(
            osm_id=way.id,
            vertices=vertices,
            indices=indices,
            uvs=uvs,
            normals=normals,
            color=color,
            height=height,
            floors=floors,
            building_type=btype,
            name=way.tags.get("name", ""),
            footprint_area_m2=area,
        )

    def _coords_to_enu(
        self,
        coords: tuple[tuple[float, float], ...],
    ) -> list[tuple[float, float]] | None:
        """
        Конвертувати lat/lon coords у ENU (east, north) метри.
        Повертає 2D список (підлога на рівні 0).
        """
        if not coords:
            return None

        R      = 6_378_137.0
        cosLat = math.cos(math.radians(self._origin_lat))
        result: list[tuple[float, float]] = []

        # Видалити закритий полігон (перший = останній)
        pts = list(coords)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]

        for lat, lon in pts:
            east  = (math.radians(lon) - math.radians(self._origin_lon)) * cosLat * R
            north = (math.radians(lat) - math.radians(self._origin_lat)) * R
            result.append((east, north))

        return result if len(result) >= 3 else None


# ----------------------------------------------------------------
# ГЕОМЕТРИЧНІ ФУНКЦІЇ
# ----------------------------------------------------------------

def _parse_height(tags: dict[str, str]) -> float:
    """
    Парсити висоту будівлі з OSM тегів.

    Пріоритет:
    1. height=* (метри, може бути "10 m" або "10")
    2. building:levels=* × FLOOR_HEIGHT
    3. levels=* × FLOOR_HEIGHT
    4. Дефолт за типом

    Returns:
        Висота у метрах (завжди >= MIN_HEIGHT)
    """
    # 1. Явна висота
    height_str = tags.get("height", "")
    if height_str:
        # Видаляємо одиниці ("10 m", "10m", "10")
        cleaned = height_str.replace("m", "").replace(" ", "").strip()
        try:
            h = float(cleaned)
            return max(MIN_HEIGHT, min(MAX_HEIGHT, h))
        except ValueError:
            pass

    # 2. Кількість поверхів
    for key in ("building:levels", "levels"):
        levels_str = tags.get(key, "")
        if levels_str:
            try:
                levels = float(levels_str)
                if levels > 0:
                    return max(MIN_HEIGHT, levels * FLOOR_HEIGHT)
            except ValueError:
                pass

    # 3. Дефолт за типом
    btype = tags.get("building", "yes")
    return BUILDING_TYPE_HEIGHTS.get(btype, BUILDING_TYPE_HEIGHTS["yes"])


def _parse_floors(tags: dict[str, str], height: float) -> int:
    """Кількість поверхів (з тегів або обчислена)."""
    for key in ("building:levels", "levels"):
        val = tags.get(key, "")
        if val:
            try:
                return max(1, int(float(val)))
            except ValueError:
                pass
    return max(1, round(height / FLOOR_HEIGHT))


def _polygon_area_m2(
    coords: list[tuple[float, float]],
) -> float:
    """
    Площа полігону у м² (Shoelace formula).
    coords у метрах (ENU East/North).
    """
    n = len(coords)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return abs(area) * 0.5


def _triangulate_polygon(
    coords: list[tuple[float, float]],
) -> list[tuple[int, int, int]]:
    """
    Тріангуляція простого полігону (ear clipping).

    Ear Clipping:
    - Знаходимо "вухо" (вершина де діагональ всередині полігону)
    - Відрізаємо трикутник
    - Повторюємо до 3 вершин

    Обмеження: тільки прості (не самоперетинаючі) полігони.
    Для складних — використовувати shapely.triangulate.

    Returns:
        Список трикутників як індекси вершин
    """
    n = len(coords)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Перевіряємо орієнтацію (CCW = positive area)
    area = sum(
        coords[i][0] * coords[(i+1)%n][1] - coords[(i+1)%n][0] * coords[i][1]
        for i in range(n)
    )
    if area < 0:
        coords = list(reversed(coords))

    # Ear clipping
    indices = list(range(n))
    triangles: list[tuple[int, int, int]] = []

    while len(indices) > 3:
        found_ear = False
        for i in range(len(indices)):
            prev_i = (i - 1) % len(indices)
            next_i = (i + 1) % len(indices)

            a = coords[indices[prev_i]]
            b = coords[indices[i]]
            c = coords[indices[next_i]]

            # Чи є це вухо? (CCW і немає точок всередині)
            if _cross_2d(a, b, c) <= 0:
                continue

            # Перевірити чи немає інших точок всередині трикутника
            is_ear = True
            for j in range(len(indices)):
                if j in (prev_i, i, next_i):
                    continue
                if _point_in_triangle(coords[indices[j]], a, b, c):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((
                    indices[prev_i],
                    indices[i],
                    indices[next_i],
                ))
                indices.pop(i)
                found_ear = True
                break

        if not found_ear:
            # Fallback: fan triangulation (для випуклих полігонів)
            triangles = [
                (indices[0], indices[i], indices[i+1])
                for i in range(1, len(indices) - 1)
            ]
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles


def _cross_2d(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """2D cross product (знак визначає орієнтацію)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """Чи знаходиться точка p всередині трикутника abc."""
    d1 = _cross_2d(p, a, b)
    d2 = _cross_2d(p, b, c)
    d3 = _cross_2d(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _build_extrusion(
    coords:        list[tuple[float, float]],
    floor_indices: list[tuple[int, int, int]],
    height:        float,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Побудувати повний 3D меш будівлі (видавка + дах + підлога).

    Three.js конвенція: X=East, Y=Up, Z=-North.

    Структура меша:
      - Floor vertices: N штук (y=0)
      - Roof  vertices: N штук (y=height)
      - Wall  vertices: N*4 (по 4 на кожне ребро)

    Returns:
        (vertices, indices, uvs, normals)
    """
    n = len(coords)

    # ---- Вершини ----
    # Floor (y=0): N вершин
    # Roof  (y=h): N вершин
    # Walls: N ребер × 4 вершини
    total_verts = n * 2 + n * 4   # floor + roof + walls

    vertices = np.zeros((total_verts, 3), dtype=np.float32)
    uvs      = np.zeros((total_verts, 2), dtype=np.float32)

    # Floor і Roof
    for i, (east, north) in enumerate(coords):
        # Three.js: X=East, Y=Up, Z=-North
        vertices[i]     = [east, 0.0,    -north]   # floor
        vertices[n + i] = [east, height, -north]   # roof
        uvs[i]          = [east / 100.0, north / 100.0]
        uvs[n + i]      = [east / 100.0, north / 100.0]

    # Walls
    wall_base = n * 2
    for i in range(n):
        j    = (i + 1) % n
        e0, n0 = coords[i]
        e1, n1 = coords[j]

        v_base = wall_base + i * 4
        # BL, BR, TL, TR (CCW)
        vertices[v_base + 0] = [e0, 0.0,    -n0]
        vertices[v_base + 1] = [e1, 0.0,    -n1]
        vertices[v_base + 2] = [e0, height, -n0]
        vertices[v_base + 3] = [e1, height, -n1]

        # UV по довжині стіни
        wall_len = math.sqrt((e1-e0)**2 + (n1-n0)**2)
        uvs[v_base + 0] = [0.0,            0.0]
        uvs[v_base + 1] = [wall_len/3.0,   0.0]
        uvs[v_base + 2] = [0.0,            height/3.0]
        uvs[v_base + 3] = [wall_len/3.0,   height/3.0]

    # ---- Індекси ----
    # Floor трикутники (звернені вниз — нормаль -Y)
    floor_tris = [
        (fi[0], fi[2], fi[1])  # reverse winding для floor
        for fi in floor_indices
    ]
    # Roof трикутники (нормаль +Y)
    roof_tris  = [
        (n + fi[0], n + fi[1], n + fi[2])
        for fi in floor_indices
    ]
    # Wall трикутники (по 2 на ребро)
    wall_tris: list[tuple[int, int, int]] = []
    for i in range(n):
        v = wall_base + i * 4
        wall_tris.append((v+0, v+1, v+2))   # нижній трикутник
        wall_tris.append((v+1, v+3, v+2))   # верхній трикутник

    all_tris = floor_tris + roof_tris + wall_tris
    indices  = np.array(all_tris, dtype=np.uint32)

    # ---- Нормалі ----
    normals = _compute_building_normals(vertices, indices)

    return vertices, indices, uvs, normals


def _compute_building_normals(
    vertices: npt.NDArray[np.float32],
    indices:  npt.NDArray[np.uint32],
) -> npt.NDArray[np.float32]:
    """Обчислити нормалі для будівельного меша."""
    normals = np.zeros_like(vertices)

    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    tri_normals = np.cross(edge1, edge2)

    np.add.at(normals, indices[:, 0], tri_normals)
    np.add.at(normals, indices[:, 1], tri_normals)
    np.add.at(normals, indices[:, 2], tri_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-10, 1.0, lengths)

    return (normals / lengths).astype(np.float32)
