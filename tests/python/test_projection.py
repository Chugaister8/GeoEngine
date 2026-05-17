"""
GeoEngine — Tile Projection Tests
"""

from __future__ import annotations

import pytest

from geoengine.geo.projection import (
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
from geoengine.geo.bbox import BBox


# ================================================================
# TileXYZ
# ================================================================

class TestTileXYZ:

    def test_valid_tile(self):
        tile = TileXYZ(x=0, y=0, z=0)
        assert tile.x == 0
        assert tile.y == 0
        assert tile.z == 0

    def test_invalid_zoom(self):
        with pytest.raises(ValueError):
            TileXYZ(x=0, y=0, z=23)  # > 22

    def test_invalid_x(self):
        with pytest.raises(ValueError):
            TileXYZ(x=-1, y=0, z=5)

    def test_invalid_y_too_large(self):
        with pytest.raises(ValueError):
            TileXYZ(x=0, y=32, z=4)   # max = 2^4 - 1 = 15

    def test_parent_z0(self):
        tile = TileXYZ(x=0, y=0, z=0)
        assert tile.parent is None

    def test_parent_z1(self):
        child  = TileXYZ(x=1, y=1, z=1)
        parent = child.parent
        assert parent is not None
        assert parent.z == 0
        assert parent.x == 0
        assert parent.y == 0

    def test_children(self):
        tile     = TileXYZ(x=0, y=0, z=0)
        children = tile.children
        assert len(children) == 4
        for child in children:
            assert child.z == 1
            assert child.parent == tile

    def test_children_coordinates(self):
        tile     = TileXYZ(x=2, y=3, z=5)
        children = tile.children
        # TL, TR, BL, BR
        assert children[0] == TileXYZ(x=4, y=6, z=6)
        assert children[1] == TileXYZ(x=5, y=6, z=6)
        assert children[2] == TileXYZ(x=4, y=7, z=6)
        assert children[3] == TileXYZ(x=5, y=7, z=6)

    def test_quadkey_z0(self):
        assert TileXYZ(x=0, y=0, z=0).to_quadkey() == ""

    def test_quadkey_round_trip(self):
        tile = TileXYZ(x=5, y=3, z=4)
        qk   = tile.to_quadkey()
        assert TileXYZ.from_quadkey(qk) == tile

    def test_quadkey_values(self):
        # z=1: 4 тайли → quadkey 0,1,2,3
        assert TileXYZ(x=0, y=0, z=1).to_quadkey() == "0"
        assert TileXYZ(x=1, y=0, z=1).to_quadkey() == "1"
        assert TileXYZ(x=0, y=1, z=1).to_quadkey() == "2"
        assert TileXYZ(x=1, y=1, z=1).to_quadkey() == "3"

    def test_frozen_immutable(self):
        tile = TileXYZ(x=1, y=2, z=3)
        with pytest.raises(Exception):
            tile.x = 5  # type: ignore


# ================================================================
# TILE ↔ BBOX
# ================================================================

class TestTileBBoxConversion:

    def test_z0_bbox_is_world(self):
        """z=0 → весь світ."""
        bbox = tile_to_bbox(TileXYZ(x=0, y=0, z=0))
        assert bbox.west  == pytest.approx(-180.0, abs=0.001)
        assert bbox.east  == pytest.approx(180.0,  abs=0.001)
        assert bbox.north > 85.0
        assert bbox.south < -85.0

    def test_bbox_width_decreases_with_zoom(self):
        """Ширина bbox зменшується вдвічі з кожним zoom."""
        z0 = tile_to_bbox(TileXYZ(x=0, y=0, z=0))
        z1 = tile_to_bbox(TileXYZ(x=0, y=0, z=1))
        assert z1.width == pytest.approx(z0.width / 2, rel=0.01)

    def test_latlon_to_tile_z0(self):
        """Будь-яка точка на z=0 → (0,0,0)."""
        tile = latlon_to_tile(lat=48.0, lon=23.0, zoom=0)
        assert tile == TileXYZ(x=0, y=0, z=0)

    def test_latlon_to_tile_round_trip(self):
        """LatLon → Tile → BBox містить вихідну точку."""
        lat, lon = 48.15, 24.50
        zoom     = 12
        tile     = latlon_to_tile(lat=lat, lon=lon, zoom=zoom)
        bbox     = tile_to_bbox(tile)
        assert bbox.contains_point(lat, lon)

    def test_latlon_to_tile_zoom_10(self):
        """Перевірка конкретного тайлу (Карпати, zoom=10)."""
        tile = latlon_to_tile(lat=48.15, lon=24.50, zoom=10)
        assert tile.z == 10
        # Перевіряємо що bbox містить вихідну точку
        bbox = tile_to_bbox(tile)
        assert (48.15, 24.50) in bbox

    def test_bbox_to_tiles_small(self):
        """Маленький bbox → кілька тайлів."""
        bbox  = BBox(west=23.0, south=48.0, east=23.5, north=48.5)
        tiles = bbox_to_tiles(bbox, zoom=10)
        assert len(tiles) > 0
        # Кожен тайл має правильний zoom
        for t in tiles:
            assert t.z == 10

    def test_bbox_covered_by_tiles(self):
        """Всі тайли разом покривають bbox."""
        bbox  = BBox(west=23.0, south=48.0, east=24.0, north=49.0)
        tiles = bbox_to_tiles(bbox, zoom=8)

        # BBox всіх тайлів разом повинен включати вихідний bbox
        tile_bboxes = [tile_to_bbox(t) for t in tiles]
        total_west  = min(b.west  for b in tile_bboxes)
        total_south = min(b.south for b in tile_bboxes)
        total_east  = max(b.east  for b in tile_bboxes)
        total_north = max(b.north for b in tile_bboxes)

        assert total_west  <= bbox.west
        assert total_south <= bbox.south
        assert total_east  >= bbox.east
        assert total_north >= bbox.north

    def test_tile_count(self):
        """tile_count = len(bbox_to_tiles)."""
        bbox  = BBox(west=23.0, south=48.0, east=24.0, north=49.0)
        zoom  = 8
        count = tile_count_for_bbox(bbox, zoom)
        tiles = bbox_to_tiles(bbox, zoom)
        assert count == len(tiles)


# ================================================================
# RESOLUTION
# ================================================================

class TestResolution:

    def test_resolution_decreases_with_zoom(self):
        """Чим більший zoom — тим менша resolution (дрібніші пікселі)."""
        tile_z5  = TileXYZ(x=0, y=0, z=5)
        tile_z10 = TileXYZ(x=0, y=0, z=10)
        assert tile_resolution_m(tile_z5) > tile_resolution_m(tile_z10)

    def test_resolution_at_equator_zoom0(self):
        """zoom=0: весь світ на 256 пікс ≈ 156км/піксель."""
        tile = TileXYZ(x=0, y=0, z=0)
        res  = tile_resolution_m(tile)
        assert res == pytest.approx(156_543, rel=0.05)

    def test_zoom_for_resolution(self):
        """zoom_for_resolution → розумний zoom рівень."""
        zoom_10m  = zoom_for_resolution(10.0)
        zoom_100m = zoom_for_resolution(100.0)
        assert zoom_10m  > zoom_100m
        assert 12 <= zoom_10m  <= 16
        assert 9  <= zoom_100m <= 13


# ================================================================
# PIXEL ↔ LATLON
# ================================================================

class TestPixelLatLon:

    def test_pixel_round_trip(self):
        """Pixel → LatLon → Pixel зберігає точність."""
        zoom = 12
        px, py = 1024.5, 2048.7

        lat, lon = pixel_to_latlon(px, py, zoom)
        px2, py2 = latlon_to_pixel(lat, lon, zoom)

        assert px2 == pytest.approx(px, abs=0.01)
        assert py2 == pytest.approx(py, abs=0.01)

    def test_origin_at_top_left(self):
        """Пікселі починаються з (0,0) у лівому верхньому куті."""
        zoom = 0
        lat, lon = pixel_to_latlon(0.0, 0.0, zoom)
        # (0,0) пікс на z=0 = NW кут карти
        assert lon < -170  # далеко на захід
        assert lat > 80    # далеко на північ
