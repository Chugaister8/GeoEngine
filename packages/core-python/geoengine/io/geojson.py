"""
GeoEngine — GeoJSON I/O
Читання та запис GeoJSON файлів.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import structlog

from ..geo.bbox import BBox
from ..dem.analysis import ContourResult

log: structlog.BoundLogger = structlog.get_logger(__name__)

# GeoJSON типи
GeoJSONGeometry  = dict[str, Any]
GeoJSONFeature   = dict[str, Any]
GeoJSONCollection = dict[str, Any]


def write_geojson(
    path:     str | Path,
    features: list[GeoJSONFeature],
    name:     str = "",
    overwrite: bool = False,
) -> Path:
    """
    Записати список GeoJSON features у файл.

    Args:
        path:     вихідний .geojson файл
        features: список Feature об'єктів
        name:     назва колекції (у properties)
        overwrite: перезаписати якщо існує

    Returns:
        Path до записаного файлу
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Файл вже існує: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    collection: GeoJSONCollection = {
        "type":     "FeatureCollection",
        "name":     name,
        "features": features,
    }

    path.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("io.geojson.write", path=str(path), features=len(features))
    return path


def read_geojson(path: str | Path) -> GeoJSONCollection:
    """Прочитати GeoJSON файл."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON файл не знайдено: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    log.debug("io.geojson.read", path=str(path))
    return data


def bbox_to_feature(bbox: BBox, properties: dict | None = None) -> GeoJSONFeature:
    """BBox → GeoJSON Feature (Polygon)."""
    return {
        "type": "Feature",
        "geometry": bbox.to_geojson(),
        "properties": properties or {},
    }


def contours_to_geojson(
    result: ContourResult,
    path:   str | Path | None = None,
) -> GeoJSONCollection:
    """
    ContourResult → GeoJSON FeatureCollection з ізолініями.

    Args:
        result: результат compute_contours()
        path:   якщо вказано — зберегти у файл

    Returns:
        GeoJSON FeatureCollection
    """
    features: list[GeoJSONFeature] = []
    for line, elev in zip(result.lines, result.elevations, strict=True):
        coords = [[lon, lat] for lat, lon in line]  # GeoJSON: [lon, lat]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "elevation": round(elev, 1),
                "elevation_unit": "meters",
            },
        })

    collection: GeoJSONCollection = {
        "type":     "FeatureCollection",
        "name":     f"contours_{result.bbox.center[0]:.3f}",
        "features": features,
    }

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("io.geojson.contours", path=str(path), lines=len(features))

    return collection


def points_to_geojson(
    points:     list[tuple[float, float]],   # (lat, lon)
    properties: list[dict] | None = None,
    path:       str | Path | None = None,
) -> GeoJSONCollection:
    """
    Список точок → GeoJSON FeatureCollection (Point).
    """
    props_list = properties or [{} for _ in points]
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],   # GeoJSON: [lon, lat]
            },
            "properties": p,
        }
        for (lat, lon), p in zip(points, props_list, strict=True)
    ]

    collection: GeoJSONCollection = {
        "type":     "FeatureCollection",
        "features": features,
    }

    if path is not None:
        Path(path).write_text(
            json.dumps(collection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return collection


def viewshed_to_geojson(
    visible_points:   list[tuple[float, float]],
    invisible_points: list[tuple[float, float]],
    observer:         tuple[float, float],
    path:             str | Path | None = None,
) -> GeoJSONCollection:
    """
    Результат viewshed → GeoJSON з двома шарами (visible/invisible).
    """
    features: list[GeoJSONFeature] = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [observer[1], observer[0]]},
            "properties": {"type": "observer"},
        }
    ]

    for lat, lon in visible_points:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"visible": True},
        })

    for lat, lon in invisible_points:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"visible": False},
        })

    collection: GeoJSONCollection = {
        "type":     "FeatureCollection",
        "features": features,
    }

    if path is not None:
        Path(path).write_text(
            json.dumps(collection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return collection
