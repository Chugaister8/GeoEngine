"""
GeoEngine — OSM REST API
Endpoints для OpenStreetMap даних.

Endpoints:
  POST /api/osm/buildings   — будівлі для bbox
  POST /api/osm/roads       — дороги для bbox
  POST /api/osm/full        — всі OSM дані
  GET  /api/osm/cache/clear — очистити OSM кеш
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from geoengine.osm.fetcher   import OverpassFetcher, OverpassError, OverpassTimeoutError
from geoengine.osm.buildings import BuildingExtruder
from geoengine.osm.roads     import RoadBuilder

router = APIRouter(prefix="/api/osm", tags=["osm"])

# Shared fetcher (disk cache між запитами)
_fetcher = OverpassFetcher()


# ----------------------------------------------------------------
# REQUEST MODELS
# ----------------------------------------------------------------

class OSMRequest(BaseModel):
    west:  float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90,  le=90)
    east:  float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90,  le=90)

    @field_validator("north")
    @classmethod
    def north_gt_south(cls, v: float, info: Any) -> float:
        if "south" in (info.data or {}) and v <= info.data["south"]:
            raise ValueError("north > south required")
        return v

    @property
    def area_deg2(self) -> float:
        return (self.east - self.west) * (self.north - self.south)

    def to_bbox(self):
        from geoengine.geo.bbox import BBox
        return BBox(
            west=self.west, south=self.south,
            east=self.east, north=self.north,
        )


class BuildingsRequest(OSMRequest):
    format:      str = Field(default="geojson", pattern="^(geojson|mesh)$")
    lod:         int = Field(default=1, ge=1, le=2)
    min_area_m2: float = Field(default=10.0, ge=0.0)


class RoadsRequest(OSMRequest):
    road_types: list[str] | None = None
    format:     str = Field(default="geojson", pattern="^(geojson|mesh)$")


# ----------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------

@router.post("/buildings")
async def get_buildings(request: BuildingsRequest) -> dict[str, Any]:
    """
    Будівлі OSM для bbox.

    Response (format=geojson): GeoJSON FeatureCollection
    Response (format=mesh):    BuildingCollection з 3D мешами
    """
    _validate_osm_bbox(request)
    bbox = request.to_bbox()

    try:
        osm_data = await _fetcher.fetch_buildings(bbox)
    except OverpassTimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail=f"Overpass timeout: {e}. Зменши bbox.",
        )
    except OverpassError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if request.format == "mesh":
        c        = bbox.center
        extruder = BuildingExtruder(
            origin_lat=c[0],
            origin_lon=c[1],
            min_area_m2=request.min_area_m2,
            lod=request.lod,
        )
        collection = extruder.extrude_all(osm_data)
        return collection.to_dict()

    # GeoJSON format
    features = []
    for way in osm_data.buildings():
        if len(way.coords) < 3:
            continue
        coords = [[lon, lat] for lat, lon in way.coords]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "osm_id":  way.id,
                "name":    way.tags.get("name", ""),
                "type":    way.tags.get("building", "yes"),
                "height":  way.tags.get("height", ""),
                "levels":  way.tags.get("building:levels", ""),
                **{k: v for k, v in way.tags.items()
                   if k not in ("name", "building", "height",
                                "building:levels")},
            },
        })

    return {
        "type":     "FeatureCollection",
        "name":     f"buildings_{bbox.center[0]:.3f}_{bbox.center[1]:.3f}",
        "bbox":     bbox.to_list(),
        "features": features,
        "count":    len(features),
        "total_buildings": osm_data.building_count,
    }


@router.post("/roads")
async def get_roads(request: RoadsRequest) -> dict[str, Any]:
    """
    Дороги OSM для bbox.
    """
    _validate_osm_bbox(request)
    bbox = request.to_bbox()

    try:
        osm_data = await _fetcher.fetch_roads(
            bbox,
            road_types=request.road_types,
        )
    except OverpassTimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))
    except OverpassError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if request.format == "mesh":
        c       = bbox.center
        builder = RoadBuilder(origin_lat=c[0], origin_lon=c[1])
        collection = builder.build_all(osm_data)
        return collection.to_dict()

    # GeoJSON
    features = []
    for way in osm_data.highways():
        if len(way.coords) < 2:
            continue
        coords = [[lon, lat] for lat, lon in way.coords]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "osm_id":  way.id,
                "name":    way.tags.get("name", ""),
                "highway": way.tags.get("highway", ""),
                "oneway":  way.tags.get("oneway", "no"),
                "maxspeed":way.tags.get("maxspeed", ""),
                "surface": way.tags.get("surface", ""),
            },
        })

    return {
        "type":     "FeatureCollection",
        "bbox":     bbox.to_list(),
        "features": features,
        "count":    len(features),
        "total_roads": len(osm_data.highways()),
    }


@router.post("/full")
async def get_full(request: OSMRequest) -> dict[str, Any]:
    """
    Всі OSM дані: будівлі + дороги + вода + природа.
    Тільки для малих bbox (< 0.01°²).
    """
    if request.area_deg2 > 0.25:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Для /full bbox занадто великий: {request.area_deg2:.4f}°². "
                "Максимум: 0.5°×0.5°."
            ),
        )

    bbox = request.to_bbox()

    try:
        osm_data = await _fetcher.fetch_full(bbox)
    except OverpassTimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))
    except OverpassError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "bbox":           bbox.to_list(),
        "node_count":     osm_data.node_count,
        "way_count":      osm_data.way_count,
        "building_count": osm_data.building_count,
        "road_count":     len(osm_data.highways()),
        "water_count":    len(osm_data.water_bodies()),
    }


@router.get("/cache/clear")
async def clear_osm_cache() -> dict[str, Any]:
    """Очистити OSM disk кеш."""
    deleted = _fetcher.clear_cache()
    return {"deleted": deleted, "status": "ok"}


# ----------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------

def _validate_osm_bbox(request: OSMRequest) -> None:
    """Перевірити bbox для OSM запитів."""
    if request.area_deg2 > 1.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"OSM bbox занадто великий: {request.area_deg2:.3f}°². "
                "Максимум: 1°×1° (~110×110 км)."
            ),
  )
