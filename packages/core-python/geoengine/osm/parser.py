"""
GeoEngine — OSM Parser
Допоміжний модуль для парсингу OSM форматів.

Підтримує:
  - OSM JSON (з Overpass API) — основний
  - OSM XML (стандартний формат) — для локальних файлів
  - PBF (Protocol Buffer) — для великих файлів (через osmium)
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import structlog

from ..geo.bbox import BBox
from .fetcher import OSMData, OSMNode, OSMWay, OSMRelation, _resolve_way_coords

log: structlog.BoundLogger = structlog.get_logger(__name__)


def parse_overpass_json(
    data:     dict | str,
    bbox:     BBox | None = None,
) -> OSMData:
    """
    Розпарсити JSON відповідь Overpass API.

    Args:
        data: dict або JSON рядок
        bbox: bbox для OSMData (якщо None → обчислюємо з nodes)

    Returns:
        OSMData
    """
    if isinstance(data, str):
        data = json.loads(data)

    elements: list[dict[str, Any]] = data.get("elements", [])

    nodes:     list[OSMNode]     = []
    ways:      list[OSMWay]      = []
    relations: list[OSMRelation] = []

    for elem in elements:
        elem_type = elem.get("type", "")
        elem_id   = int(elem.get("id", 0))
        tags      = {str(k): str(v) for k, v in elem.get("tags", {}).items()}

        if elem_type == "node":
            lat = elem.get("lat")
            lon = elem.get("lon")
            if lat is not None and lon is not None:
                nodes.append(OSMNode(
                    id=elem_id,
                    lat=float(lat),
                    lon=float(lon),
                    tags=tags,
                ))

        elif elem_type == "way":
            node_ids = tuple(int(n) for n in elem.get("nodes", []))
            ways.append(OSMWay(
                id=elem_id,
                node_ids=node_ids,
                tags=tags,
            ))

        elif elem_type == "relation":
            members = tuple(elem.get("members", []))
            relations.append(OSMRelation(
                id=elem_id,
                members=members,
                tags=tags,
            ))

    # Обчислити bbox якщо не задано
    if bbox is None and nodes:
        lats = [n.lat for n in nodes]
        lons = [n.lon for n in nodes]
        bbox = BBox(
            west=min(lons), south=min(lats),
            east=max(lons), north=max(lats),
        )
    elif bbox is None:
        bbox = BBox(west=0, south=0, east=0, north=0)

    osm_data = OSMData(
        bbox=bbox,
        nodes=nodes,
        ways=ways,
        relations=relations,
    )
    _resolve_way_coords(osm_data)

    return osm_data


def parse_osm_xml(
    source: str | Path | bytes,
    bbox:   BBox | None = None,
) -> OSMData:
    """
    Розпарсити OSM XML файл або рядок.

    Формат OSM XML:
    <osm version="0.6">
      <node id="1" lat="48.0" lon="23.0">
        <tag k="name" v="MyNode"/>
      </node>
      <way id="2">
        <nd ref="1"/>
        <tag k="highway" v="primary"/>
      </way>
    </osm>

    Args:
        source: шлях до файлу, XML рядок або bytes
        bbox:   bbox для OSMData

    Returns:
        OSMData
    """
    if isinstance(source, Path):
        tree = ET.parse(source)
        root = tree.getroot()
    elif isinstance(source, bytes):
        root = ET.fromstring(source)
    else:
        root = ET.fromstring(source)

    nodes:     list[OSMNode]     = []
    ways:      list[OSMWay]      = []
    relations: list[OSMRelation] = []

    # Парсинг nodes
    for elem in root.findall("node"):
        elem_id = int(elem.get("id", 0))
        lat     = elem.get("lat")
        lon     = elem.get("lon")

        if lat is None or lon is None:
            continue

        tags = {
            tag.get("k", ""): tag.get("v", "")
            for tag in elem.findall("tag")
            if tag.get("k")
        }
        nodes.append(OSMNode(
            id=elem_id,
            lat=float(lat),
            lon=float(lon),
            tags=tags,
        ))

    # Парсинг ways
    for elem in root.findall("way"):
        elem_id  = int(elem.get("id", 0))
        node_ids = tuple(int(nd.get("ref", 0)) for nd in elem.findall("nd"))
        tags     = {
            tag.get("k", ""): tag.get("v", "")
            for tag in elem.findall("tag")
            if tag.get("k")
        }
        ways.append(OSMWay(
            id=elem_id,
            node_ids=node_ids,
            tags=tags,
        ))

    # Парсинг relations
    for elem in root.findall("relation"):
        elem_id = int(elem.get("id", 0))
        members = tuple({
            "type": m.get("type", ""),
            "ref":  int(m.get("ref", 0)),
            "role": m.get("role", ""),
        } for m in elem.findall("member"))
        tags = {
            tag.get("k", ""): tag.get("v", "")
            for tag in elem.findall("tag")
            if tag.get("k")
        }
        relations.append(OSMRelation(
            id=elem_id,
            members=members,
            tags=tags,
        ))

    # Bbox з bounds елементу або з nodes
    if bbox is None:
        bounds = root.find("bounds")
        if bounds is not None:
            bbox = BBox(
                west=float(bounds.get("minlon", -180)),
                south=float(bounds.get("minlat", -90)),
                east=float(bounds.get("maxlon", 180)),
                north=float(bounds.get("maxlat", 90)),
            )
        elif nodes:
            lats = [n.lat for n in nodes]
            lons = [n.lon for n in nodes]
            bbox = BBox(
                west=min(lons), south=min(lats),
                east=max(lons), north=max(lats),
            )
        else:
            bbox = BBox(west=0, south=0, east=0, north=0)

    osm_data = OSMData(
        bbox=bbox,
        nodes=nodes,
        ways=ways,
        relations=relations,
    )
    _resolve_way_coords(osm_data)

    log.info(
        "osm.xml.parsed",
        nodes=len(nodes),
        ways=len(ways),
        relations=len(relations),
    )
    return osm_data


def load_osm_file(path: str | Path, bbox: BBox | None = None) -> OSMData:
    """
    Завантажити OSM файл автоматично визначаючи формат.

    Підтримує: .osm (XML), .json (Overpass JSON), .osm.json

    Args:
        path: шлях до файлу
        bbox: bbox (якщо None — обчислюється автоматично)

    Returns:
        OSMData

    Raises:
        ValueError: невідомий формат
        FileNotFoundError: файл не існує
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OSM файл не знайдено: {path}")

    suffix = "".join(path.suffixes).lower()
    log.info("osm.load", path=str(path), format=suffix)

    if suffix in (".osm", ".xml"):
        return parse_osm_xml(path, bbox)
    elif suffix in (".json", ".osm.json", ".geojson"):
        data = json.loads(path.read_text(encoding="utf-8"))
        return parse_overpass_json(data, bbox)
    else:
        raise ValueError(
            f"Невідомий формат OSM файлу: {suffix}. "
            "Підтримується: .osm, .xml, .json"
      )
