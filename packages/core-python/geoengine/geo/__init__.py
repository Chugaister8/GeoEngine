"""
GeoEngine — geo пакет
Експортує всі публічні типи та функції геопросторового шару.
"""

from .bbox import BBox
from .coords import (
    LLH,
    ECEF,
    ENU,
    WebMercator,
    DEG2RAD,
    RAD2DEG,
    llh_to_ecef,
    ecef_to_llh,
    llh_to_enu,
    enu_to_llh,
    llh_to_webmercator,
    webmercator_to_llh,
    haversine_distance,
    vincenty_distance,
    bearing,
)
from .projection import (
    TileXYZ,
    TILE_SIZE,
    tile_to_bbox,
    latlon_to_tile,
    bbox_to_tiles,
    tile_count_for_bbox,
    tile_resolution_m,
    zoom_for_resolution,
    latlon_to_pixel,
    pixel_to_latlon,
)

__all__ = [
    # BBox
    "BBox",
    # Точки
    "LLH",
    "ECEF",
    "ENU",
    "WebMercator",
    # Константи
    "DEG2RAD",
    "RAD2DEG",
    # Конвертації
    "llh_to_ecef",
    "ecef_to_llh",
    "llh_to_enu",
    "enu_to_llh",
    "llh_to_webmercator",
    "webmercator_to_llh",
    # Відстані
    "haversine_distance",
    "vincenty_distance",
    "bearing",
    # Тайли
    "TileXYZ",
    "TILE_SIZE",
    "tile_to_bbox",
    "latlon_to_tile",
    "bbox_to_tiles",
    "tile_count_for_bbox",
    "tile_resolution_m",
    "zoom_for_resolution",
    "latlon_to_pixel",
    "pixel_to_latlon",
]
