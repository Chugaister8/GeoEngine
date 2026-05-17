"""
GeoEngine — osm пакет
OpenStreetMap завантаження, парсинг та 3D генерація.
"""

from .fetcher import (
    OverpassFetcher,
    OverpassQuery,
    OverpassError,
    OverpassTimeoutError,
    OSMData,
    OSMNode,
    OSMWay,
    OSMRelation,
    OSMElementType,
)
from .buildings import (
    BuildingExtruder,
    BuildingMesh,
    BuildingCollection,
    BUILDING_TYPE_HEIGHTS,
    BUILDING_COLORS,
)
from .roads import (
    RoadBuilder,
    RoadMesh,
    RoadCollection,
    ROAD_WIDTHS,
    ROAD_COLORS,
)
from .parser import (
    parse_overpass_json,
    parse_osm_xml,
    load_osm_file,
)

__all__ = [
    # Fetcher
    "OverpassFetcher",
    "OverpassQuery",
    "OverpassError",
    "OverpassTimeoutError",
    "OSMData",
    "OSMNode",
    "OSMWay",
    "OSMRelation",
    "OSMElementType",
    # Buildings
    "BuildingExtruder",
    "BuildingMesh",
    "BuildingCollection",
    "BUILDING_TYPE_HEIGHTS",
    "BUILDING_COLORS",
    # Roads
    "RoadBuilder",
    "RoadMesh",
    "RoadCollection",
    "ROAD_WIDTHS",
    "ROAD_COLORS",
    # Parser
    "parse_overpass_json",
    "parse_osm_xml",
    "load_osm_file",
]
