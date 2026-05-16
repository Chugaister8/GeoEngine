"""
GeoEngine — Python Core
Геопросторовий рушій: DEM, Mesh, OSM, Scene, Analysis.

Версія відповідає semver: MAJOR.MINOR.PATCH
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__  = "GeoEngine Contributors"
__license__ = "MIT"

# Зручний імпорт найчастіших типів з верхнього рівня
from .geo import (
    BBox,
    LLH,
    ECEF,
    ENU,
    WebMercator,
    TileXYZ,
    llh_to_ecef,
    ecef_to_llh,
    llh_to_enu,
    enu_to_llh,
    haversine_distance,
    vincenty_distance,
    bearing,
    tile_to_bbox,
    latlon_to_tile,
    bbox_to_tiles,
)

__all__ = [
    "__version__",
    "BBox",
    "LLH",
    "ECEF",
    "ENU",
    "WebMercator",
    "TileXYZ",
    "llh_to_ecef",
    "ecef_to_llh",
    "llh_to_enu",
    "enu_to_llh",
    "haversine_distance",
    "vincenty_distance",
    "bearing",
    "tile_to_bbox",
    "latlon_to_tile",
    "bbox_to_tiles",
]
