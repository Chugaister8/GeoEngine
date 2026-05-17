"""
GeoEngine — BBox Tests
Повне покриття BBox класу.
"""

from __future__ import annotations

import math
import pytest

from geoengine.geo.bbox import BBox


# ================================================================
# СТВОРЕННЯ ТА ВАЛІДАЦІЯ
# ================================================================

class TestBBoxCreation:

    def test_valid_bbox(self):
        bbox = BBox(west=22.0, south=47.5, east=25.0, north=49.5)
        assert bbox.west  == 22.0
        assert bbox.south == 47.5
        assert bbox.east  == 25.0
        assert bbox.north == 49.5

    def test_invalid_south_gt_north(self):
        with pytest.raises(ValueError, match="south.*north"):
            BBox(west=0, south=50.0, east=10, north=40.0)

    def test_invalid_lat_out_of_range(self):
        with pytest.raises(ValueError, match="south"):
            BBox(west=0, south=-91.0, east=10, north=0)

        with pytest.raises(ValueError, match="north"):
            BBox(west=0, south=0, east=10, north=91.0)

    def test_invalid_lon_out_of_range(self):
        with pytest.raises(ValueError, match="west"):
            BBox(west=-181.0, south=0, east=10, north=10)

    def test_frozen_immutable(self):
        bbox = BBox(west=0, south=0, east=10, north=10)
        with pytest.raises(Exception):
            bbox.west = 5  # type: ignore

    def test_from_list(self):
        bbox = BBox.from_list([22.0, 47.5, 25.0, 49.5])
        assert bbox.west  == 22.0
        assert bbox.south == 47.5

    def test_from_list_wrong_length(self):
        with pytest.raises(ValueError, match="4"):
            BBox.from_list([1, 2, 3])

    def test_world(self):
        w = BBox.world()
        assert w.west  == -180.0
        assert w.east  ==  180.0
        assert w.south ==  -90.0
        assert w.north ==   90.0

    def test_ukraine(self):
        ua = BBox.ukraine()
        assert ua.west  < 23.0
        assert ua.east  > 39.0
        assert ua.south < 45.0
        assert ua.north > 51.0


# ================================================================
# ВЛАСТИВОСТІ
# ================================================================

class TestBBoxProperties:

    def test_width(self):
        bbox = BBox(west=10.0, south=0, east=20.0, north=10)
        assert bbox.width == pytest.approx(10.0)

    def test_height(self):
        bbox = BBox(west=0, south=10.0, east=10, north=20.0)
        assert bbox.height == pytest.approx(10.0)

    def test_center(self):
        bbox = BBox(west=10.0, south=20.0, east=30.0, north=40.0)
        lat, lon = bbox.center
        assert lat == pytest.approx(30.0)
        assert lon == pytest.approx(20.0)

    def test_area_deg2(self):
        bbox = BBox(west=0, south=0, east=1.0, north=1.0)
        assert bbox.area_deg2 == pytest.approx(1.0)

    def test_area_m2_approximately(self):
        # 1°×1° на екваторі ≈ 111320² м²
        bbox = BBox(west=0, south=0, east=1.0, north=1.0)
        expected = 111_320 ** 2
        assert bbox.area_m2 == pytest.approx(expected, rel=0.05)

    def test_corners(self):
        bbox = BBox(west=10.0, south=20.0, east=30.0, north=40.0)
        sw, se, ne, nw = bbox.corners
        assert sw == (20.0, 10.0)  # (lat, lon)
        assert ne == (40.0, 30.0)

    def test_not_crosses_antimeridian(self):
        bbox = BBox(west=10.0, south=0, east=20.0, north=10)
        assert not bbox.crosses_antimeridian

    def test_crosses_antimeridian(self):
        # bbox що перетинає антимеридіан: west > east
        bbox = BBox(west=170.0, south=0, east=-170.0, north=10)
        assert bbox.crosses_antimeridian


# ================================================================
# CONTAINS / INTERSECTS
# ================================================================

class TestBBoxContains:

    def test_contains_point_inside(self):
        bbox = BBox(west=10.0, south=20.0, east=30.0, north=40.0)
        assert bbox.contains_point(30.0, 20.0)

    def test_contains_point_on_boundary(self):
        bbox = BBox(west=10.0, south=20.0, east=30.0, north=40.0)
        assert bbox.contains_point(20.0, 10.0)  # corner

    def test_not_contains_point_outside(self):
        bbox = BBox(west=10.0, south=20.0, east=30.0, north=40.0)
        assert not bbox.contains_point(50.0, 20.0)  # north of bbox

    def test_contains_operator(self):
        bbox = BBox(west=10.0, south=20.0, east=30.0, north=40.0)
        assert (30.0, 20.0) in bbox
        assert (50.0, 20.0) not in bbox

    def test_intersects_overlapping(self):
        a = BBox(west=0, south=0, east=10, north=10)
        b = BBox(west=5, south=5, east=15, north=15)
        assert a.intersects(b)
        assert b.intersects(a)

    def test_not_intersects_separate(self):
        a = BBox(west=0,  south=0, east=10, north=10)
        b = BBox(west=20, south=0, east=30, north=10)
        assert not a.intersects(b)

    def test_intersects_touching(self):
        a = BBox(west=0, south=0, east=10, north=10)
        b = BBox(west=10, south=0, east=20, north=10)
        # Touching edge — залежить від реалізації (включно)
        # У нашому випадку west <= east, тому touching = intersects
        assert a.intersects(b)


# ================================================================
# ОПЕРАЦІЇ
# ================================================================

class TestBBoxOperations:

    def test_intersection(self):
        a = BBox(west=0,  south=0,  east=10, north=10)
        b = BBox(west=5,  south=5,  east=15, north=15)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.west  == 5.0
        assert inter.south == 5.0
        assert inter.east  == 10.0
        assert inter.north == 10.0

    def test_no_intersection(self):
        a = BBox(west=0,  south=0, east=5,  north=5)
        b = BBox(west=10, south=0, east=15, north=5)
        assert a.intersection(b) is None

    def test_union(self):
        a = BBox(west=0, south=0, east=10, north=10)
        b = BBox(west=5, south=5, east=15, north=15)
        u = a.union(b)
        assert u.west  == 0.0
        assert u.south == 0.0
        assert u.east  == 15.0
        assert u.north == 15.0

    def test_expand(self):
        bbox     = BBox(west=10, south=10, east=20, north=20)
        expanded = bbox.expand(1.0)
        assert expanded.west  == pytest.approx(9.0)
        assert expanded.south == pytest.approx(9.0)
        assert expanded.east  == pytest.approx(21.0)
        assert expanded.north == pytest.approx(21.0)

    def test_expand_clamps_to_world(self):
        bbox     = BBox(west=-179, south=-89, east=179, north=89)
        expanded = bbox.expand(5.0)
        assert expanded.west  >= -180.0
        assert expanded.south >= -90.0
        assert expanded.east  <= 180.0
        assert expanded.north <= 90.0

    def test_subdivide_2x2(self):
        bbox  = BBox(west=0, south=0, east=10, north=10)
        parts = bbox.subdivide(nx=2, ny=2)
        assert len(parts) == 4

        # Перевіряємо що вони покривають весь оригінальний bbox
        all_west  = min(p.west  for p in parts)
        all_south = min(p.south for p in parts)
        all_east  = max(p.east  for p in parts)
        all_north = max(p.north for p in parts)

        assert all_west  == pytest.approx(0.0)
        assert all_south == pytest.approx(0.0)
        assert all_east  == pytest.approx(10.0)
        assert all_north == pytest.approx(10.0)

    def test_subdivide_3x3(self):
        bbox  = BBox(west=0, south=0, east=9, north=9)
        parts = bbox.subdivide(nx=3, ny=3)
        assert len(parts) == 9

    def test_from_center(self):
        bbox = BBox.from_center(lat=48.0, lon=23.0, radius_m=1000)
        lat_c, lon_c = bbox.center
        assert lat_c == pytest.approx(48.0, abs=0.001)
        assert lon_c == pytest.approx(23.0, abs=0.001)
        # Розмір приблизно 2×1000м
        assert bbox.area_m2 > 0

    def test_from_points(self):
        points = [(48.0, 23.0), (49.0, 24.0), (47.5, 22.5)]
        bbox   = BBox.from_points(points)
        assert bbox.south == 47.5
        assert bbox.north == 49.0
        assert bbox.west  == 22.5
        assert bbox.east  == 24.0

    def test_from_points_empty(self):
        with pytest.raises(ValueError, match="одна"):
            BBox.from_points([])


# ================================================================
# SERIALIZATION
# ================================================================

class TestBBoxSerialization:

    def test_to_list(self):
        bbox = BBox(west=1, south=2, east=3, north=4)
        assert bbox.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_to_tuple(self):
        bbox = BBox(west=1, south=2, east=3, north=4)
        assert bbox.to_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_to_wkt(self):
        bbox = BBox(west=10, south=20, east=30, north=40)
        wkt  = bbox.to_wkt()
        assert wkt.startswith("POLYGON")
        assert "10" in wkt
        assert "20" in wkt

    def test_to_geojson(self):
        bbox    = BBox(west=10, south=20, east=30, north=40)
        geojson = bbox.to_geojson()
        assert geojson["type"] == "Polygon"
        coords  = geojson["coordinates"][0]
        assert len(coords) == 5   # замкнений полігон
        assert coords[0] == [10.0, 20.0]   # SW corner [lon, lat]

    def test_round_trip(self):
        original = BBox(west=22.15, south=44.39, east=40.22, north=52.38)
        restored = BBox.from_list(original.to_list())
        assert restored == original

    def test_repr(self):
        bbox = BBox(west=22.0, south=47.5, east=25.0, north=49.5)
        r    = repr(bbox)
        assert "22.0000" in r
        assert "47.5000" in r
