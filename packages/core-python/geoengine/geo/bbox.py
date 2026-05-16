"""
GeoEngine — BoundingBox
Географічний обмежуючий прямокутник у WGS84.
Незмінний (frozen dataclass) — безпечний для кешування та хешування.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, NamedTuple

# ----------------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------------

EARTH_RADIUS_M: float = 6_371_000.0
WGS84_A: float = 6_378_137.0


# ----------------------------------------------------------------
# BOUNDING BOX
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BBox:
    """
    Географічний обмежуючий прямокутник (WGS84, EPSG:4326).

    Конвенція:
        west  = мін. longitude (-180..180)
        south = мін. latitude  (-90..90)
        east  = макс. longitude
        north = макс. latitude

    Антимеридіан: якщо west > east — bbox перетинає антимеридіан.

    Examples:
        >>> bbox = BBox(west=22.0, south=47.5, east=40.2, north=52.3)
        >>> (48.0, 30.0) in bbox
        True
    """
    west:  float
    south: float
    east:  float
    north: float

    # ---- Валідація ----

    def __post_init__(self) -> None:
        if not (-90.0 <= self.south <= 90.0):
            raise ValueError(f"south={self.south} поза діапазоном [-90, 90]")
        if not (-90.0 <= self.north <= 90.0):
            raise ValueError(f"north={self.north} поза діапазоном [-90, 90]")
        if self.south > self.north:
            raise ValueError(
                f"south={self.south} > north={self.north}: "
                "використовуй BBox(south=..., north=...) правильно"
            )
        if not (-180.0 <= self.west <= 180.0):
            raise ValueError(f"west={self.west} поза діапазоном [-180, 180]")
        if not (-180.0 <= self.east <= 180.0):
            raise ValueError(f"east={self.east} поза діапазоном [-180, 180]")

    # ---- Фабричні методи ----

    @classmethod
    def from_center(
        cls,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> "BBox":
        """
        Створити BBox з центру та радіусу в метрах.

        Args:
            lat:      широта центру (градуси)
            lon:      довгота центру (градуси)
            radius_m: радіус у метрах

        Returns:
            BBox що охоплює коло радіусу radius_m навколо (lat, lon)
        """
        delta_lat = math.degrees(radius_m / EARTH_RADIUS_M)
        # longitude degrees скорочуються на cos(lat)
        lat_rad = math.radians(lat)
        delta_lon = math.degrees(
            radius_m / (EARTH_RADIUS_M * math.cos(lat_rad))
        ) if abs(lat) < 89.9 else 180.0

        return cls(
            west=max(-180.0, lon - delta_lon),
            south=max(-90.0,  lat - delta_lat),
            east=min(180.0,   lon + delta_lon),
            north=min(90.0,   lat + delta_lat),
        )

    @classmethod
    def from_points(cls, points: list[tuple[float, float]]) -> "BBox":
        """
        Мінімальний BBox що охоплює всі точки (lat, lon).

        Args:
            points: список (lat, lon) кортежів

        Raises:
            ValueError: якщо points порожній
        """
        if not points:
            raise ValueError("Потрібна хоча б одна точка")

        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return cls(
            west=min(lons),
            south=min(lats),
            east=max(lons),
            north=max(lats),
        )

    @classmethod
    def world(cls) -> "BBox":
        """Весь світ."""
        return cls(west=-180.0, south=-90.0, east=180.0, north=90.0)

    @classmethod
    def ukraine(cls) -> "BBox":
        """Приблизний BBox України."""
        return cls(west=22.15, south=44.39, east=40.22, north=52.38)

    # ---- Властивості ----

    @property
    def width(self) -> float:
        """Ширина у градусах longitude."""
        if self.crosses_antimeridian:
            return (180.0 - self.west) + (self.east + 180.0)
        return self.east - self.west

    @property
    def height(self) -> float:
        """Висота у градусах latitude."""
        return self.north - self.south

    @property
    def center(self) -> tuple[float, float]:
        """Центр як (lat, lon)."""
        center_lat = (self.south + self.north) / 2.0
        if self.crosses_antimeridian:
            center_lon = self.west + self.width / 2.0
            if center_lon > 180.0:
                center_lon -= 360.0
        else:
            center_lon = (self.west + self.east) / 2.0
        return center_lat, center_lon

    @property
    def crosses_antimeridian(self) -> bool:
        """Чи перетинає bbox антимеридіан (180° / -180°)."""
        return self.west > self.east

    @property
    def area_deg2(self) -> float:
        """Площа у квадратних градусах (приблизно)."""
        return self.width * self.height

    @property
    def area_m2(self) -> float:
        """
        Площа у квадратних метрах (наближення через рівнокутну проекцію).
        Точна тільки для невеликих bbox; для великих — використовуй pyproj.
        """
        lat_rad    = math.radians((self.south + self.north) / 2.0)
        width_m    = math.radians(self.width)  * EARTH_RADIUS_M * math.cos(lat_rad)
        height_m   = math.radians(self.height) * EARTH_RADIUS_M
        return abs(width_m * height_m)

    @property
    def corners(self) -> tuple[
        tuple[float, float],   # SW
        tuple[float, float],   # SE
        tuple[float, float],   # NE
        tuple[float, float],   # NW
    ]:
        """Чотири кути як (lat, lon): SW, SE, NE, NW."""
        return (
            (self.south, self.west),  # SW
            (self.south, self.east),  # SE
            (self.north, self.east),  # NE
            (self.north, self.west),  # NW
        )

    # ---- Операції ----

    def contains_point(self, lat: float, lon: float) -> bool:
        """Чи знаходиться точка (lat, lon) всередині bbox."""
        if not (self.south <= lat <= self.north):
            return False
        if self.crosses_antimeridian:
            return lon >= self.west or lon <= self.east
        return self.west <= lon <= self.east

    def __contains__(self, point: tuple[float, float]) -> bool:
        """Синтаксичний цукор: (lat, lon) in bbox."""
        return self.contains_point(point[0], point[1])

    def intersects(self, other: "BBox") -> bool:
        """Чи перетинаються два bbox."""
        # Перевірка по latitude (проста)
        if self.north < other.south or other.north < self.south:
            return False
        # Перевірка по longitude (складніша через антимеридіан)
        if self.crosses_antimeridian or other.crosses_antimeridian:
            # Спрощення: розбиваємо на два нормальних bbox
            for a in self._split_antimeridian():
                for b in other._split_antimeridian():
                    if not (a.east < b.west or b.east < a.west):
                        return True
            return False
        return not (self.east < other.west or other.east < self.west)

    def intersection(self, other: "BBox") -> "BBox | None":
        """
        Перетин двох bbox.

        Returns:
            BBox перетину або None якщо не перетинаються.
        """
        if not self.intersects(other):
            return None
        return BBox(
            west=max(self.west,  other.west),
            south=max(self.south, other.south),
            east=min(self.east,  other.east),
            north=min(self.north, other.north),
        )

    def union(self, other: "BBox") -> "BBox":
        """Мінімальний bbox що охоплює обидва."""
        return BBox(
            west=min(self.west,  other.west),
            south=min(self.south, other.south),
            east=max(self.east,  other.east),
            north=max(self.north, other.north),
        )

    def expand(self, degrees: float) -> "BBox":
        """Розширити bbox на degrees у всіх напрямках."""
        return BBox(
            west=max(-180.0, self.west  - degrees),
            south=max(-90.0,  self.south - degrees),
            east=min(180.0,   self.east  + degrees),
            north=min(90.0,   self.north + degrees),
        )

    def expand_meters(self, meters: float) -> "BBox":
        """Розширити bbox на meters у всіх напрямках."""
        delta = math.degrees(meters / EARTH_RADIUS_M)
        return self.expand(delta)

    def subdivide(self, nx: int = 2, ny: int = 2) -> list["BBox"]:
        """
        Розбити bbox на nx×ny рівних частин.

        Args:
            nx: кількість колонок (по longitude)
            ny: кількість рядків  (по latitude)

        Returns:
            Список nx*ny bbox, впорядкований рядками (зверху вниз)
        """
        lon_step = self.width  / nx
        lat_step = self.height / ny
        result: list[BBox] = []
        for row in range(ny):
            for col in range(nx):
                result.append(BBox(
                    west=self.west  + col * lon_step,
                    south=self.south + row * lat_step,
                    east=self.west  + (col + 1) * lon_step,
                    north=self.south + (row + 1) * lat_step,
                ))
        return result

    def to_wkt(self) -> str:
        """WKT POLYGON представлення (для PostGIS, QGIS)."""
        w, s, e, n = self.west, self.south, self.east, self.north
        return (
            f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
        )

    def to_geojson(self) -> dict:
        """GeoJSON Polygon представлення."""
        w, s, e, n = self.west, self.south, self.east, self.north
        return {
            "type": "Polygon",
            "coordinates": [[
                [w, s], [e, s], [e, n], [w, n], [w, s]
            ]]
        }

    def to_list(self) -> list[float]:
        """[west, south, east, north] — формат для більшості API."""
        return [self.west, self.south, self.east, self.north]

    def to_tuple(self) -> tuple[float, float, float, float]:
        """(west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)

    @classmethod
    def from_list(cls, bbox: list[float] | tuple) -> "BBox":
        """З [west, south, east, north]."""
        if len(bbox) != 4:
            raise ValueError(f"Очікується 4 елементи, отримано {len(bbox)}")
        return cls(
            west=float(bbox[0]),
            south=float(bbox[1]),
            east=float(bbox[2]),
            north=float(bbox[3]),
        )

    # ---- Приватні хелпери ----

    def _split_antimeridian(self) -> list["BBox"]:
        """Розбити на два bbox якщо перетинає антимеридіан."""
        if not self.crosses_antimeridian:
            return [self]
        return [
            BBox(west=self.west,  south=self.south, east=180.0,  north=self.north),
            BBox(west=-180.0, south=self.south, east=self.east, north=self.north),
        ]

    def __repr__(self) -> str:
        return (
            f"BBox(W={self.west:.4f}, S={self.south:.4f}, "
            f"E={self.east:.4f}, N={self.north:.4f})"
      )
