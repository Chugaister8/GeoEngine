"""
GeoEngine — Python Core v0.1.0
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__  = "GeoEngine Contributors"
__license__ = "MIT"

# Geo primitives
from .geo import (
    BBox, LLH, ECEF, ENU, WebMercator, TileXYZ,
    llh_to_ecef, ecef_to_llh,
    llh_to_enu, enu_to_llh,
    haversine_distance, vincenty_distance, bearing,
    tile_to_bbox, latlon_to_tile, bbox_to_tiles,
)

# Utils
from .utils import Vec2, Vec3, Quat, Mat4, get_logger

# Scene
from .scene import Scene, SceneNode, NodeType, CameraState, Layer, LayerManager

__all__ = [
    "__version__",
    # Geo
    "BBox", "LLH", "ECEF", "ENU", "WebMercator", "TileXYZ",
    "llh_to_ecef", "ecef_to_llh", "llh_to_enu", "enu_to_llh",
    "haversine_distance", "vincenty_distance", "bearing",
    "tile_to_bbox", "latlon_to_tile", "bbox_to_tiles",
    # Utils
    "Vec2", "Vec3", "Quat", "Mat4", "get_logger",
    # Scene
    "Scene", "SceneNode", "NodeType", "CameraState",
    "Layer", "LayerManager",
]
