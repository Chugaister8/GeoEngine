"""
GeoEngine — OSM Roads
Конвертація OSM доріг у 3D геометрію.

Алгоритм:
  OSMWay (лінія) → буфер (ширина дороги) → меш з UV

Ширини доріг (метри) за типом:
  motorway:  12m (2 смуги по 6м)
  primary:    8m
  secondary:  6m
  tertiary:   5m
  residential:4m
  service:    3m
  path:       1.5m

LOD:
  LOD0: повна геометрія + розмітка (UV)
  LOD1: спрощена геометрія (менше вершин)
  LOD2: просто лінії (для далеких планів)
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

# Ширини доріг у метрах (половина — для буферизації з двох боків)
ROAD_WIDTHS: Final[dict[str, float]] = {
    "motorway":        12.0,
    "motorway_link":    6.0,
    "trunk":           10.0,
    "trunk_link":       5.0,
    "primary":          8.0,
    "primary_link":     4.0,
    "secondary":        6.5,
    "secondary_link":   3.0,
    "tertiary":         5.5,
    "tertiary_link":    2.5,
    "unclassified":     4.5,
    "residential":      4.0,
    "living_street":    3.5,
    "service":          3.0,
    "pedestrian":       4.0,
    "track":            3.0,
    "footway":          1.5,
    "cycleway":         2.0,
    "path":             1.5,
    "steps":            2.0,
    "default":          4.0,
}

# Підняття дороги над рельєфом (щоб не z-fighting)
ROAD_ELEVATION_OFFSET: Final[float] = 0.3   # метри

# Кольори доріг (RGB 0-1)
ROAD_COLORS: Final[dict[str, tuple[float, float, float]]] = {
    "motorway":   (0.95, 0.70, 0.30),
    "trunk":      (0.95, 0.75, 0.40),
    "primary":    (0.95, 0.85, 0.50),
    "secondary":  (0.85, 0.85, 0.60),
    "tertiary":   (0.80, 0.80, 0.70),
    "residential":(0.90, 0.90, 0.90),
    "footway":    (0.85, 0.75, 0.65),
    "cycleway":   (0.50, 0.75, 0.95),
    "default":    (0.85, 0.85, 0.85),
}


# ----------------------------------------------------------------
# ТИПИ РЕЗУЛЬТАТУ
# ----------------------------------------------------------------

@dataclass(slots=True)
class RoadMesh:
    """3D меш однієї дороги."""
    osm_id:    int
    vertices:  npt.NDArray[np.float32]   # (N, 3)
    indices:   npt.NDArray[np.uint32]    # (M, 3)
    uvs:       npt.NDArray[np.float32]   # (N, 2)
    normals:   npt.NDArray[np.float32]   # (N, 3)
    color:     tuple[float, float, float]
    road_type: str
    width_m:   float
    length_m:  float
    name:      str = ""

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    def to_dict(self) -> dict:
        import base64

        def b64(a: npt.NDArray) -> str:
            return base64.b64encode(a.tobytes()).decode("ascii")

        return {
            "type":          "road_mesh",
            "osm_id":        self.osm_id,
            "vertex_count":  self.vertex_count,
            "triangle_count": self.triangle_count,
            "road_type":     self.road_type,
            "width_m":       self.width_m,
            "length_m":      round(self.length_m, 1),
            "name":          self.name,
            "color":         list(self.color),
            "buffers": {
                "vertices": b64(self.vertices),
                "indices":  b64(self.indices),
                "uvs":      b64(self.uvs),
                "normals":  b64(self.normals),
            },
        }


@dataclass(slots=True)
class RoadCollection:
    """Колекція дорожніх мешів."""
    meshes:      list[RoadMesh]
    origin_lat:  float
    origin_lon:  float
    total_roads: int
    skipped:     int = 0

    @property
    def vertex_count(self) -> int:
        return sum(m.vertex_count for m in self.meshes)

    @property
    def triangle_count(self) -> int:
        return sum(m.triangle_count for m in self.meshes)

    def to_dict(self) -> dict:
        return {
            "type":           "road_collection",
            "count":          len(self.meshes),
            "total":          self.total_roads,
            "skipped":        self.skipped,
            "origin":         {"lat": self.origin_lat, "lon": self.origin_lon},
            "vertex_count":   self.vertex_count,
            "triangle_count": self.triangle_count,
            "roads":          [m.to_dict() for m in self.meshes],
        }


# ----------------------------------------------------------------
# ROAD GEOMETRY BUILDER
# ----------------------------------------------------------------

class RoadBuilder:
    """
    Конвертує OSM дороги у 3D меші.

    Алгоритм:
    1. Отримати polyline coords
    2. Визначити ширину за типом
    3. Для кожного сегменту → rectangle (4 вершини)
    4. З'єднати сегменти (miter joints)
    5. Повернути RoadMesh

    Usage:
        builder = RoadBuilder(origin_lat=48.0, origin_lon=23.0)
        collection = builder.build_all(osm_data)
    """

    def __init__(
        self,
        origin_lat:  float,
        origin_lon:  float,
        lod:         Literal[0, 1, 2] = 0,
        elevation:   float = ROAD_ELEVATION_OFFSET,
    ) -> None:
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._lod        = lod
        self._elevation  = elevation

    # ---- Публічний API ----

    def build_all(self, osm_data: OSMData) -> RoadCollection:
        """Побудувати всі дороги з OSMData."""
        highways = osm_data.highways()
        log.info(
            "roads.build.start",
            count=len(highways),
            lod=self._lod,
        )

        meshes:  list[RoadMesh] = []
        skipped: int = 0

        for way in highways:
            try:
                mesh = self._build_one(way)
                if mesh is not None:
                    meshes.append(mesh)
                else:
                    skipped += 1
            except Exception as exc:
                log.debug(
                    "roads.build.skip",
                    osm_id=way.id,
                    error=str(exc)[:80],
                )
                skipped += 1

        log.info(
            "roads.build.done",
            built=len(meshes),
            skipped=skipped,
            verts=sum(m.vertex_count for m in meshes),
        )

        return RoadCollection(
            meshes=meshes,
            origin_lat=self._origin_lat,
            origin_lon=self._origin_lon,
            total_roads=len(highways),
            skipped=skipped,
        )

    # ---- Приватні методи ----

    def _build_one(self, way: OSMWay) -> RoadMesh | None:
        """Побудувати один дорожній меш."""
        if len(way.coords) < 2:
            return None

        road_type = way.tags.get("highway", "default")
        width_m   = ROAD_WIDTHS.get(road_type, ROAD_WIDTHS["default"])
        half_w    = width_m * 0.5
        color     = ROAD_COLORS.get(road_type, ROAD_COLORS["default"])
        name      = way.tags.get("name", "")

        # Конвертувати coords у ENU
        enu_pts = self._coords_to_enu(way.coords)
        if len(enu_pts) < 2:
            return None

        # Довжина дороги
        length_m = sum(
            math.sqrt((enu_pts[i+1][0]-enu_pts[i][0])**2
                    + (enu_pts[i+1][1]-enu_pts[i][1])**2)
            for i in range(len(enu_pts) - 1)
        )

        if length_m < 1.0:
            return None

        # Побудувати ribbon (стрічку) вздовж polyline
        vertices, indices, uvs = _build_road_ribbon(
            enu_pts, half_w, self._elevation
        )

        # Нормалі (дорога горизонтальна → нормаль вгору)
        normals = np.tile(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (len(vertices), 1),
        )

        return RoadMesh(
            osm_id=way.id,
            vertices=vertices,
            indices=indices,
            uvs=uvs,
            normals=normals,
            color=color,
            road_type=road_type,
            width_m=width_m,
            length_m=length_m,
            name=name,
        )

    def _coords_to_enu(
        self,
        coords: tuple[tuple[float, float], ...],
    ) -> list[tuple[float, float]]:
        """Конвертувати lat/lon → ENU (east, north) метри."""
        R      = 6_378_137.0
        cosLat = math.cos(math.radians(self._origin_lat))
        result: list[tuple[float, float]] = []

        for lat, lon in coords:
            east  = (math.radians(lon) - math.radians(self._origin_lon)) * cosLat * R
            north = (math.radians(lat) - math.radians(self._origin_lat)) * R
            result.append((east, north))

        return result


# ----------------------------------------------------------------
# ROAD RIBBON GEOMETRY
# ----------------------------------------------------------------

def _build_road_ribbon(
    pts:       list[tuple[float, float]],
    half_w:    float,
    elevation: float,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float32],
]:
    """
    Побудувати ribbon геометрію вздовж polyline.

    Алгоритм miter joins:
    - Для кожної вершини обчислюємо miter вектор
    - Miter = нормалізований bisector між двома сегментами
    - Ширина miter обмежена (уникаємо artifact на гострих кутах)

    Returns:
        (vertices, indices, uvs) — Three.js формат
    """
    n = len(pts)

    # Обчислити normals для кожного сегменту
    seg_normals: list[tuple[float, float]] = []
    for i in range(n - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            seg_normals.append((0.0, 1.0))
        else:
            seg_normals.append((-dy / length, dx / length))

    # Miter normals для кожної вершини
    miter_normals: list[tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            miter_normals.append(seg_normals[0])
        elif i == n - 1:
            miter_normals.append(seg_normals[-1])
        else:
            n1 = seg_normals[i-1]
            n2 = seg_normals[i]
            # Bisector
            mx = n1[0] + n2[0]
            my = n1[1] + n2[1]
            ml = math.sqrt(mx*mx + my*my)
            if ml < 1e-6:
                miter_normals.append(n1)
            else:
                # Miter scale (обмежуємо до 3×)
                cos_a = n1[0]*n2[0] + n1[1]*n2[1]
                miter_scale = min(3.0, 1.0 / max(0.1, (1.0 + cos_a) * 0.5))
                miter_normals.append((
                    mx / ml * miter_scale,
                    my / ml * miter_scale,
                ))

    # Побудувати vertices
    # Для кожної точки polyline — 2 вершини (left + right)
    num_verts = n * 2
    vertices  = np.zeros((num_verts, 3), dtype=np.float32)
    uvs       = np.zeros((num_verts, 2), dtype=np.float32)

    # Кумулятивна відстань для UV
    dist: float = 0.0
    for i, (east, north) in enumerate(pts):
        if i > 0:
            pe, pn = pts[i-1]
            dist += math.sqrt((east-pe)**2 + (north-pn)**2)

        nx, ny = miter_normals[i]

        # Left verts: x = east - ny*half_w, z = -(north + nx*half_w)
        # Right verts: x = east + ny*half_w, z = -(north - nx*half_w)
        left_e  = east  - ny * half_w
        left_n  = north + nx * half_w
        right_e = east  + ny * half_w
        right_n = north - nx * half_w

        # Three.js: X=East, Y=Up(elevation), Z=-North
        vertices[i*2 + 0] = [left_e,  elevation, -left_n]
        vertices[i*2 + 1] = [right_e, elevation, -right_n]

        uvs[i*2 + 0] = [0.0, dist / (half_w * 2)]
        uvs[i*2 + 1] = [1.0, dist / (half_w * 2)]

    # Індекси (два трикутники на кожен сегмент)
    num_tris = (n - 1) * 2
    indices  = np.zeros((num_tris, 3), dtype=np.uint32)

    for i in range(n - 1):
        tl = i * 2       # top-left
        tr = i * 2 + 1   # top-right
        bl = (i+1) * 2   # bottom-left
        br = (i+1) * 2 + 1  # bottom-right

        indices[i*2 + 0] = [tl, bl, tr]
        indices[i*2 + 1] = [tr, bl, br]

    return vertices, indices, uvs
