"""
GeoEngine — Tile Projection
XYZ тайловий адресний простір ↔ BBox ↔ Pixel координати.
Критична частина: всі тайли терейну адресуються через цей модуль.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Iterator

from .bbox import BBox
from .coords import LLH, WebMercator, llh_to_webmercator

# ----------------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------------

TILE_SIZE:  Final[int]   = 256            # пікселів на тайл
_ORIGIN_M:  Final[float] = 20_037_508.342789244  # max WebMercator X/Y

# ----------------------------------------------------------------
# TILE XYZ
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TileXYZ:
    """
    Адреса тайлу у схемі XYZ (Slippy Map).

    z=0: 1 тайл — весь світ
    z=1: 2×2 = 4 тайли
    z=N: 2^N × 2^N тайлів
    """
    x: int
    y: int
    z: int   # zoom 0..22

    def __post_init__(self) -> None:
        if not (0 <= self.z <= 22):
            raise ValueError(f"zoom={self.z} поза [0, 22]")
        max_idx = (1 << self.z) - 1   # 2^z - 1
        if not (0 <= self.x <= max_idx):
            raise ValueError(f"x={self.x} поза [0, {max_idx}] для z={self.z}")
        if not (0 <= self.y <= max_idx):
            raise ValueError(f"y={self.y} поза [0, {max_idx}] для z={self.z}")

    @property
    def parent(self) -> "TileXYZ | None":
        """Батьківський тайл (z-1). None для z=0."""
        if self.z == 0:
            return None
        return TileXYZ(x=self.x >> 1, y=self.y >> 1, z=self.z - 1)

    @property
    def children(self) -> tuple["TileXYZ", "TileXYZ", "TileXYZ", "TileXYZ"]:
        """Чотири дочірніх тайли (z+1): TL, TR, BL, BR."""
        x2, y2, z1 = self.x * 2, self.y * 2, self.z + 1
        return (
            TileXYZ(x=x2,     y=y2,     z=z1),  # TL
            TileXYZ(x=x2 + 1, y=y2,     z=z1),  # TR
            TileXYZ(x=x2,     y=y2 + 1, z=z1),  # BL
            TileXYZ(x=x2 + 1, y=y2 + 1, z=z1),  # BR
        )

    @property
    def siblings(self) -> list["TileXYZ"]:
        """Всі 4 тайли одного рівня з тим самим батьком."""
        p = self.parent
        if p is None:
            return [TileXYZ(0, 0, 0)]
        return list(p.children)

    def to_quadkey(self) -> str:
        """
        Конвертувати в Quadkey (Microsoft Bing Maps формат).
        Кожен символ (0-3) визначає квадрант на кожному рівні.
        """
        key = []
        for level in range(self.z, 0, -1):
            digit = 0
            mask = 1 << (level - 1)
            if self.x & mask:
                digit += 1
            if self.y & mask:
                digit += 2
            key.append(str(digit))
        return "".join(key)

    @classmethod
    def from_quadkey(cls, quadkey: str) -> "TileXYZ":
        """Створити TileXYZ з Quadkey рядка."""
        z = len(quadkey)
        x = y = 0
        for level, char in enumerate(quadkey):
            mask = 1 << (z - 1 - level)
            digit = int(char)
            if digit & 1:
                x |= mask
            if digit & 2:
                y |= mask
        return cls(x=x, y=y, z=z)

    def __repr__(self) -> str:
        return f"Tile(x={self.x}, y={self.y}, z={self.z})"


# ----------------------------------------------------------------
# TILE ↔ BBOX КОНВЕРТАЦІЯ
# ----------------------------------------------------------------

def tile_to_bbox(tile: TileXYZ) -> BBox:
    """
    Отримати географічний BBox тайлу (WGS84).

    Це стандартна WebMercator → WGS84 конвертація,
    яка лежить в основі будь-якої тайлової карти.
    """
    n = 1 << tile.z  # 2^z

    west  = tile.x / n * 360.0 - 180.0
    east  = (tile.x + 1) / n * 360.0 - 180.0

    # WebMercator Y → latitude
    north = _merc_y_to_lat(tile.y,     n)
    south = _merc_y_to_lat(tile.y + 1, n)

    return BBox(west=west, south=south, east=east, north=north)


def latlon_to_tile(lat: float, lon: float, zoom: int) -> TileXYZ:
    """
    Знайти тайл що містить географічну точку (lat, lon) на zoom рівні.

    Args:
        lat:  широта (-85.051129..85.051129)
        lon:  довгота (-180..180)
        zoom: рівень зуму (0..22)

    Returns:
        TileXYZ адреса тайлу
    """
    lat_clamped = max(-85.051129, min(85.051129, lat))
    n = 1 << zoom  # 2^zoom

    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat_clamped)
    y = int((1.0 - math.log(
        math.tan(lat_r) + 1.0 / math.cos(lat_r)
    ) / math.pi) / 2.0 * n)

    # Clip до валідного діапазону
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))

    return TileXYZ(x=x, y=y, z=zoom)


def bbox_to_tiles(bbox: BBox, zoom: int) -> list[TileXYZ]:
    """
    Отримати всі тайли на zoom що покривають bbox.

    Args:
        bbox: географічний BBox
        zoom: рівень деталізації

    Returns:
        Список TileXYZ, впорядкований зліва-направо, зверху-вниз

    Warning:
        На великих zoom та великому bbox може повертати мільйони тайлів.
        Перевіряй розмір перед викликом: tile_count_for_bbox(bbox, zoom).
    """
    tl = latlon_to_tile(bbox.north, bbox.west, zoom)  # top-left
    br = latlon_to_tile(bbox.south, bbox.east, zoom)  # bottom-right

    tiles: list[TileXYZ] = []
    for y in range(tl.y, br.y + 1):
        for x in range(tl.x, br.x + 1):
            tiles.append(TileXYZ(x=x, y=y, z=zoom))
    return tiles


def tile_count_for_bbox(bbox: BBox, zoom: int) -> int:
    """Кількість тайлів що покривають bbox на zoom рівні."""
    tl = latlon_to_tile(bbox.north, bbox.west, zoom)
    br = latlon_to_tile(bbox.south, bbox.east, zoom)
    return (br.x - tl.x + 1) * (br.y - tl.y + 1)


def tile_resolution_m(tile: TileXYZ) -> float:
    """
    Роздільна здатність тайлу в метрах на піксель.
    Залежить від широти (WebMercator стискає на полюсах).

    Returns:
        Середня роздільна здатність для даного тайлу (м/піксель)
    """
    bbox = tile_to_bbox(tile)
    center_lat = (bbox.south + bbox.north) / 2.0
    # Стандартна формула WebMercator resolution
    return (
        2.0 * math.pi * 6_378_137.0
        * math.cos(math.radians(center_lat))
        / (TILE_SIZE * (1 << tile.z))
    )


def zoom_for_resolution(resolution_m: float, lat: float = 0.0) -> int:
    """
    Визначити оптимальний zoom для бажаної роздільної здатності.

    Args:
        resolution_m: бажана роздільна здатність (м/піксель)
        lat:          широта (впливає на точність)

    Returns:
        Zoom рівень (0..22)
    """
    zoom_f = math.log2(
        2.0 * math.pi * 6_378_137.0
        * math.cos(math.radians(lat))
        / (resolution_m * TILE_SIZE)
    )
    return max(0, min(22, round(zoom_f)))


# ----------------------------------------------------------------
# PIXEL ↔ LATLON
# ----------------------------------------------------------------

def latlon_to_pixel(
    lat: float,
    lon: float,
    zoom: int,
) -> tuple[float, float]:
    """
    Географічна точка → піксельні координати на zoom рівні.
    Глобальні пікселі (не відносно тайлу).

    Returns:
        (px, py) — піксельні координати
    """
    n = float(1 << zoom)
    lat_r = math.radians(max(-85.051129, min(85.051129, lat)))

    px = (lon + 180.0) / 360.0 * n * TILE_SIZE
    py = (1.0 - math.log(
        math.tan(lat_r) + 1.0 / math.cos(lat_r)
    ) / math.pi) / 2.0 * n * TILE_SIZE

    return px, py


def pixel_to_latlon(
    px: float,
    py: float,
    zoom: int,
) -> tuple[float, float]:
    """
    Глобальні піксельні координати → (lat, lon).

    Returns:
        (lat, lon) у градусах
    """
    n = float(1 << zoom) * TILE_SIZE
    lon = px / n * 360.0 - 180.0
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - 2.0 * py / n)))
    )
    return lat, lon


# ----------------------------------------------------------------
# ПРИВАТНІ ХЕЛПЕРИ
# ----------------------------------------------------------------

def _merc_y_to_lat(y: int, n: int) -> float:
    """WebMercator тайловий Y → latitude (градуси)."""
    return math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
  )
