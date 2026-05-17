"""
GeoEngine — Scene
Головний контейнер сцени.

Scene = кореневий SceneNode + LayerManager + CameraState
      + глобальні налаштування (освітлення, атмосфера, fog).

Є центральним об'єктом Python API:
    scene = Scene()
    scene.add_terrain(dem_tile)
    scene.add_buildings(osm_data)
    scene.camera.lat = 48.25
    scene.render()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from .node    import SceneNode, NodeType, Transform
from .layer   import Layer, LayerManager
from .camera  import CameraState, CameraBookmark, CameraAnimation
from ..geo.bbox    import BBox
from ..geo.coords  import LLH
from ..dem.loader  import DEMTile
from ..utils.math3d import Vec3

log: structlog.BoundLogger = structlog.get_logger(__name__)


# ----------------------------------------------------------------
# LIGHTING CONFIG
# ----------------------------------------------------------------

@dataclass
class SunLight:
    """Напрямок та інтенсивність сонця."""
    direction:  tuple[float, float, float] = (0.5, 0.8, 0.3)
    intensity:  float = 1.2
    color:      tuple[float, float, float] = (1.0, 0.98, 0.92)
    ambient:    float = 0.35
    cast_shadows: bool = True

    def to_dict(self) -> dict:
        return {
            "direction":    list(self.direction),
            "intensity":    self.intensity,
            "color":        list(self.color),
            "ambient":      self.ambient,
            "cast_shadows": self.cast_shadows,
        }


@dataclass
class FogSettings:
    """Налаштування атмосферного туману."""
    color:   tuple[float, float, float] = (0.8, 0.9, 1.0)
    density: float = 0.000015
    start:   float = 10_000.0
    end:     float = 500_000.0
    height:  float = 5_000.0   # висота де туман зникає

    def to_dict(self) -> dict:
        return {
            "color":   list(self.color),
            "density": self.density,
            "start":   self.start,
            "end":     self.end,
            "height":  self.height,
        }


@dataclass
class AtmosphereSettings:
    """Параметри атмосферного розсіювання."""
    rayleigh_coeff: tuple[float, float, float] = (5.5e-6, 13.0e-6, 22.4e-6)
    mie_coeff:      float = 21e-6
    mie_dir:        float = 0.758
    sun_intensity:  float = 22.0
    star_brightness: float = 0.5

    def to_dict(self) -> dict:
        return {
            "rayleigh_coeff": list(self.rayleigh_coeff),
            "mie_coeff":      self.mie_coeff,
            "mie_dir":        self.mie_dir,
            "sun_intensity":  self.sun_intensity,
            "star_brightness": self.star_brightness,
        }


# ----------------------------------------------------------------
# TERRAIN NODE
# ----------------------------------------------------------------

class TerrainNode(SceneNode):
    """Вузол що містить один DEM тайл."""

    def __init__(
        self,
        dem_tile:  DEMTile,
        lod_level: int = 0,
        name:      str = "",
    ) -> None:
        super().__init__(
            name=name or f"terrain_{dem_tile.bbox.center[0]:.3f}",
            node_type=NodeType.TERRAIN,
        )
        self.dem_tile  = dem_tile
        self.lod_level = lod_level
        self.metadata["source"] = dem_tile.source

    def to_dict(self, include_children: bool = True) -> dict:
        d = super().to_dict(include_children)
        d["lod_level"]    = self.lod_level
        d["bbox"]         = self.dem_tile.bbox.to_list()
        d["min_elevation"] = self.dem_tile.min_elevation
        d["max_elevation"] = self.dem_tile.max_elevation
        return d


# ----------------------------------------------------------------
# SCENE
# ----------------------------------------------------------------

class Scene:
    """
    Головний об'єкт сцени GeoEngine.

    Управляє:
    - Ієрархією SceneNode
    - Шарами (LayerManager)
    - Камерою та анімаціями
    - Глобальними налаштуваннями (освітлення, fog)
    - Серіалізацією для WebSocket/REST

    Usage:
        scene = Scene(name="Carpathians")

        # Камера
        scene.camera.lat = 48.15
        scene.camera.lon = 24.50
        scene.camera.alt = 8000

        # Терейн
        tile = await dem_manager.fetch(bbox)
        scene.add_terrain(tile)

        # Будівлі
        osm = await overpass.fetch_buildings(bbox)
        scene.add_buildings(osm)

        # Серіалізація
        data = scene.to_dict()
    """

    def __init__(
        self,
        name:   str = "GeoEngine Scene",
        origin: LLH | None = None,
    ) -> None:
        self.name     = name
        self.origin   = origin or LLH(lat=48.0, lon=23.0, alt=0.0)

        # Кореневий вузол
        self._root    = SceneNode(name="__root__", node_type=NodeType.GROUP)

        # Шари
        self.layers   = LayerManager.with_defaults()

        # Камера
        self.camera   = CameraState(
            lat=self.origin.lat,
            lon=self.origin.lon,
            alt=5000.0,
        )

        # Освітлення та атмосфера
        self.sun        = SunLight()
        self.fog        = FogSettings()
        self.atmosphere = AtmosphereSettings()

        # Час доби [0..1] (0=північ, 0.5=полудень)
        self.time_of_day: float = 0.5

        # Анімація
        self._camera_anim: CameraAnimation | None = None
        self._bookmarks:   list[CameraBookmark]   = []

        # Статистика
        self._created_at = time.time()

        log.info("scene.created", name=name, origin=str(origin))

    # ---- Властивості ----

    @property
    def root(self) -> SceneNode:
        return self._root

    @property
    def node_count(self) -> int:
        return sum(1 for _ in self._root.iter_all()) - 1  # не рахуємо root

    @property
    def bbox(self) -> BBox | None:
        """BBox що охоплює всі terrain nodes."""
        terrain_nodes = self._root.find_by_type(NodeType.TERRAIN)
        if not terrain_nodes:
            return None
        bboxes = [
            n.dem_tile.bbox  # type: ignore[attr-defined]
            for n in terrain_nodes
            if isinstance(n, TerrainNode)
        ]
        if not bboxes:
            return None
        result = bboxes[0]
        for b in bboxes[1:]:
            result = result.union(b)
        return result

    # ---- Додавання вузлів ----

    def add(
        self,
        node:     SceneNode,
        layer_id: str = "annotations",
    ) -> "Scene":
        """
        Додати довільний вузол до сцени.

        Args:
            node:     вузол
            layer_id: шар (default: annotations)

        Returns:
            self (для chaining)
        """
        self._root.add_child(node)
        if layer_id in self.layers:
            self.layers[layer_id].add(node)
        return self

    def add_terrain(
        self,
        dem_tile:  DEMTile,
        lod_level: int = 0,
        name:      str = "",
    ) -> TerrainNode:
        """
        Додати DEM тайл як TerrainNode.

        Returns:
            Створений TerrainNode
        """
        node = TerrainNode(dem_tile=dem_tile, lod_level=lod_level, name=name)
        self._root.add_child(node)
        if "base_terrain" in self.layers:
            self.layers["base_terrain"].add(node)
        log.debug(
            "scene.add_terrain",
            name=node.name,
            bbox=str(dem_tile.bbox),
            lod=lod_level,
        )
        return node

    def add_buildings(
        self,
        osm_data:  Any,   # OSMData
        origin:    LLH | None = None,
        layer_id:  str = "osm_buildings",
    ) -> SceneNode:
        """
        Додати будівлі з OSM даних.

        Args:
            osm_data: OSMData з fetcher
            origin:   ENU origin (None = center of osm_data.bbox)
            layer_id: шар

        Returns:
            GroupNode з будівлями
        """
        from ..osm.buildings import BuildingExtruder

        if origin is None:
            c = osm_data.bbox.center
            origin = LLH(lat=c[0], lon=c[1])

        extruder   = BuildingExtruder(origin_lat=origin.lat, origin_lon=origin.lon)
        collection = extruder.extrude_all(osm_data)

        group = SceneNode(
            name=f"buildings_{len(collection.meshes)}",
            node_type=NodeType.BUILDING,
        )
        group.metadata["building_count"] = len(collection.meshes)
        group.metadata["vertex_count"]   = collection.vertex_count

        self._root.add_child(group)
        if layer_id in self.layers:
            self.layers[layer_id].add(group)

        log.info(
            "scene.add_buildings",
            count=len(collection.meshes),
            verts=collection.vertex_count,
        )
        return group

    def add_roads(
        self,
        osm_data: Any,
        origin:   LLH | None = None,
        layer_id: str = "osm_roads",
    ) -> SceneNode:
        """Додати дороги з OSM даних."""
        from ..osm.roads import RoadBuilder

        if origin is None:
            c = osm_data.bbox.center
            origin = LLH(lat=c[0], lon=c[1])

        builder    = RoadBuilder(origin_lat=origin.lat, origin_lon=origin.lon)
        collection = builder.build_all(osm_data)

        group = SceneNode(
            name=f"roads_{len(collection.meshes)}",
            node_type=NodeType.ROAD,
        )
        group.metadata["road_count"]   = len(collection.meshes)
        group.metadata["vertex_count"] = collection.vertex_count

        self._root.add_child(group)
        if layer_id in self.layers:
            self.layers[layer_id].add(group)

        return group

    # ---- Камера ----

    def fly_to(
        self,
        lat:      float,
        lon:      float,
        alt:      float,
        heading:  float = 0.0,
        pitch:    float = -30.0,
    ) -> "Scene":
        """Встановити камеру на позицію."""
        self.camera = CameraState(
            lat=lat, lon=lon, alt=alt,
            heading=heading, pitch=pitch,
        )
        return self

    def add_bookmark(
        self,
        name:        str,
        description: str = "",
    ) -> CameraBookmark:
        """Зберегти поточну позицію камери як bookmark."""
        bm = CameraBookmark(
            name=name,
            state=CameraState(
                lat=self.camera.lat,
                lon=self.camera.lon,
                alt=self.camera.alt,
                heading=self.camera.heading,
                pitch=self.camera.pitch,
            ),
            description=description,
        )
        self._bookmarks.append(bm)
        return bm

    def get_bookmark(self, name: str) -> CameraBookmark | None:
        for bm in self._bookmarks:
            if bm.name == name:
                return bm
        return None

    # ---- Час доби ----

    def set_time(self, hours: float) -> "Scene":
        """
        Встановити час доби.

        Args:
            hours: години [0..24]
        """
        self.time_of_day = (hours % 24) / 24.0
        # Оновлюємо напрямок сонця
        angle_rad = (self.time_of_day - 0.25) * 2 * 3.14159
        import math
        self.sun.direction = (
            math.cos(angle_rad),
            math.sin(angle_rad) * 0.8 + 0.2,
            math.sin(angle_rad) * 0.3,
        )
        return self

    # ---- Серіалізація ----

    def to_dict(self) -> dict:
        """Повна серіалізація сцени для WS/REST."""
        return {
            "name":        self.name,
            "origin":      {
                "lat": self.origin.lat,
                "lon": self.origin.lon,
                "alt": self.origin.alt,
            },
            "camera":      self.camera.to_dict(),
            "sun":         self.sun.to_dict(),
            "fog":         self.fog.to_dict(),
            "atmosphere":  self.atmosphere.to_dict(),
            "time_of_day": self.time_of_day,
            "layers":      self.layers.to_dict(),
            "node_count":  self.node_count,
            "bbox":        self.bbox.to_list() if self.bbox else None,
            "bookmarks":   [bm.to_dict() for bm in self._bookmarks],
        }

    def to_json(self) -> str:
        """Серіалізація у JSON рядок."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return (
            f"Scene(name={self.name!r}, "
            f"nodes={self.node_count}, "
            f"layers={self.layers.layer_count})"
        )
