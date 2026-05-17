"""
GeoEngine — OSM Tests
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from geoengine.osm.fetcher import (
    OverpassQuery,
    OSMData,
    OSMNode,
    OSMWay,
)
from geoengine.osm.parser import parse_overpass_json, parse_osm_xml
from geoengine.osm.buildings import (
    BuildingExtruder,
    _parse_height,
    _parse_floors,
    _polygon_area_m2,
    _triangulate_polygon,
    FLOOR_HEIGHT,
    MIN_HEIGHT,
)
from geoengine.osm.roads import RoadBuilder, ROAD_WIDTHS
from geoengine.geo.bbox import BBox


# ================================================================
# OVERPASS QUERY BUILDER
# ================================================================

class TestOverpassQuery:

    def test_buildings_query(self, bbox_small):
        ql = OverpassQuery().bbox(bbox_small).buildings().build()
        assert "building" in ql
        assert "out:json"  in ql
        assert str(bbox_small.south) in ql

    def test_highways_query(self, bbox_small):
        ql = OverpassQuery().bbox(bbox_small).highways(["primary", "secondary"]).build()
        assert "highway" in ql
        assert "primary" in ql

    def test_missing_bbox_raises(self):
        with pytest.raises(ValueError, match="BBox"):
            OverpassQuery().buildings().build()

    def test_empty_parts_raises(self, bbox_small):
        with pytest.raises(ValueError, match="частин"):
            OverpassQuery().bbox(bbox_small).build()

    def test_timeout_in_query(self, bbox_small):
        ql = OverpassQuery().bbox(bbox_small).buildings().timeout(90).build()
        assert "timeout:90" in ql

    def test_bbox_format(self, bbox_small):
        """bbox у форматі south,west,north,east."""
        ql = OverpassQuery().bbox(bbox_small).buildings().build()
        b  = bbox_small
        # south,west,north,east
        expected = f"{b.south},{b.west},{b.north},{b.east}"
        assert expected in ql

    def test_water_query(self, bbox_small):
        ql = OverpassQuery().bbox(bbox_small).water().build()
        assert "water" in ql

    def test_natural_query(self, bbox_small):
        ql = OverpassQuery().bbox(bbox_small).natural(["wood", "water"]).build()
        assert "natural" in ql
        assert "wood" in ql

    def test_custom_query(self, bbox_small):
        ql = OverpassQuery().bbox(bbox_small)\
            .custom('way["shop"="supermarket"]').build()
        assert "supermarket" in ql


# ================================================================
# OSM PARSER
# ================================================================

class TestOSMParser:

    def test_parse_overpass_json(self, osm_overpass_response, bbox_small):
        data = parse_overpass_json(osm_overpass_response, bbox_small)
        assert isinstance(data, OSMData)
        assert data.node_count == 7
        assert data.way_count  == 3
        assert data.building_count == 2

    def test_nodes_have_coords(self, osm_data):
        for node in osm_data.nodes:
            assert -90 <= node.lat <= 90
            assert -180 <= node.lon <= 180

    def test_ways_have_coords_resolved(self, osm_data):
        """Coords у ways заповнені після resolve."""
        buildings = osm_data.buildings()
        for b in buildings:
            assert len(b.coords) >= 3

    def test_building_detection(self, osm_data):
        assert osm_data.building_count == 2

    def test_highway_detection(self, osm_data):
        highways = osm_data.highways()
        assert len(highways) == 1
        assert highways[0].tags["highway"] == "primary"

    def test_get_node_by_id(self, osm_data):
        node = osm_data.get_node(1)
        assert node is not None
        assert node.id == 1

    def test_get_nonexistent_node(self, osm_data):
        assert osm_data.get_node(99999) is None

    def test_ways_with_tag(self, osm_data):
        residential = osm_data.ways_with_tag("building", "residential")
        assert len(residential) == 1
        assert residential[0].id == 100

    def test_parse_osm_xml(self, tmp_path):
        """Парсинг OSM XML формату."""
        xml_content = """<?xml version="1.0"?>
<osm version="0.6">
  <bounds minlat="48.0" minlon="23.0" maxlat="48.1" maxlon="23.1"/>
  <node id="1" lat="48.001" lon="23.001">
    <tag k="name" v="Test"/>
  </node>
  <way id="2">
    <nd ref="1"/>
    <tag k="highway" v="primary"/>
  </way>
</osm>"""
        path = tmp_path / "test.osm"
        path.write_text(xml_content, encoding="utf-8")

        data = parse_osm_xml(path)
        assert data.node_count == 1
        assert data.way_count  == 1
        assert data.nodes[0].tags.get("name") == "Test"


# ================================================================
# BUILDINGS
# ================================================================

class TestParseHeight:

    def test_explicit_height(self):
        assert _parse_height({"height": "15"}) == pytest.approx(15.0)

    def test_height_with_unit(self):
        assert _parse_height({"height": "15 m"}) == pytest.approx(15.0)

    def test_levels(self):
        # 3 поверхи × 3.2м = 9.6м
        h = _parse_height({"building:levels": "3"})
        assert h == pytest.approx(3 * FLOOR_HEIGHT)

    def test_fallback_to_type(self):
        h = _parse_height({"building": "church"})
        assert h == 15.0   # church default

    def test_min_height_enforced(self):
        h = _parse_height({"height": "0.5"})
        assert h >= MIN_HEIGHT

    def test_max_height_enforced(self):
        h = _parse_height({"height": "9999"})
        assert h <= 828.0  # Бурдж-Халіфа


class TestPolygonArea:

    def test_square_area(self):
        """Квадрат 10м×10м → 100м²."""
        coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        area   = _polygon_area_m2(coords)
        assert area == pytest.approx(100.0)

    def test_triangle_area(self):
        """Прямокутний трикутник 6×8 → 24м²."""
        coords = [(0.0, 0.0), (8.0, 0.0), (0.0, 6.0)]
        area   = _polygon_area_m2(coords)
        assert area == pytest.approx(24.0)

    def test_single_point_zero_area(self):
        assert _polygon_area_m2([(0, 0)]) == 0.0


class TestTriangulate:

    def test_triangle(self):
        coords = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        tris   = _triangulate_polygon(coords)
        assert len(tris) == 1
        assert tris[0] == (0, 1, 2)

    def test_square_gives_two_triangles(self):
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        tris   = _triangulate_polygon(coords)
        assert len(tris) == 2

    def test_pentagon(self):
        import math
        n      = 5
        coords = [(math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)) for i in range(n)]
        tris   = _triangulate_polygon(coords)
        assert len(tris) == n - 2   # завжди n-2 трикутники

    def test_all_vertices_covered(self):
        coords = [(0,0), (2,0), (2,1), (1,2), (0,1)]
        tris   = _triangulate_polygon(coords)
        used   = set()
        for t in tris:
            used.update(t)
        assert 0 in used
        assert len(coords) - 1 in used


class TestBuildingExtruder:

    def test_extrude_all(self, osm_data, bbox_small):
        extruder   = BuildingExtruder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = extruder.extrude_all(osm_data)

        assert len(collection.meshes) > 0
        assert collection.total_buildings == osm_data.building_count

    def test_mesh_has_geometry(self, osm_data, bbox_small):
        extruder   = BuildingExtruder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = extruder.extrude_all(osm_data)

        for mesh in collection.meshes:
            assert mesh.vertex_count   > 0
            assert mesh.triangle_count > 0
            assert mesh.vertices.shape == (mesh.vertex_count, 3)
            assert mesh.indices.shape  == (mesh.triangle_count, 3)

    def test_mesh_height_from_tags(self, osm_data, bbox_small):
        extruder   = BuildingExtruder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = extruder.extrude_all(osm_data)

        # Будівля 100 має height=9
        mesh_100 = next(
            (m for m in collection.meshes if m.osm_id == 100), None
        )
        assert mesh_100 is not None
        assert mesh_100.height == pytest.approx(9.0)

    def test_mesh_serialization(self, osm_data, bbox_small):
        extruder   = BuildingExtruder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = extruder.extrude_all(osm_data)

        for mesh in collection.meshes:
            d = mesh.to_dict()
            assert d["type"]           == "building_mesh"
            assert d["vertex_count"]   == mesh.vertex_count
            assert d["triangle_count"] == mesh.triangle_count
            assert "vertices" in d["buffers"]
            assert "indices"  in d["buffers"]


# ================================================================
# ROADS
# ================================================================

class TestRoadBuilder:

    def test_build_all(self, osm_data, bbox_small):
        builder    = RoadBuilder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = builder.build_all(osm_data)
        assert len(collection.meshes) > 0

    def test_road_width_from_type(self, osm_data, bbox_small):
        builder    = RoadBuilder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = builder.build_all(osm_data)

        primary = next((m for m in collection.meshes if m.road_type == "primary"), None)
        if primary:
            assert primary.width_m == ROAD_WIDTHS["primary"]

    def test_road_has_geometry(self, osm_data, bbox_small):
        builder    = RoadBuilder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = builder.build_all(osm_data)

        for mesh in collection.meshes:
            assert mesh.vertex_count   > 0
            assert mesh.triangle_count > 0
            assert mesh.length_m       > 0

    def test_road_serialization(self, osm_data, bbox_small):
        builder    = RoadBuilder(
            origin_lat=bbox_small.center[0],
            origin_lon=bbox_small.center[1],
        )
        collection = builder.build_all(osm_data)

        for mesh in collection.meshes:
            d = mesh.to_dict()
            assert d["type"]     == "road_mesh"
            assert "vertices" in d["buffers"]
