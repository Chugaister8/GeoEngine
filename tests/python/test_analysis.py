"""
GeoEngine — DEM Analysis Tests
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from geoengine.dem.analysis import (
    compute_slope,
    compute_aspect,
    compute_hillshade,
    compute_contours,
    compute_profile,
)


# ================================================================
# SLOPE
# ================================================================

class TestSlope:

    def test_flat_terrain_zero_slope(self, bbox_small):
        """Плоский терейн → slope ≈ 0°."""
        from rasterio.transform import from_bounds
        from geoengine.dem.loader import DEMTile

        flat_data = np.full((32, 32), 500.0, dtype=np.float32)
        transform = from_bounds(
            bbox_small.west, bbox_small.south,
            bbox_small.east, bbox_small.north,
            32, 32,
        )
        tile = DEMTile(
            data=flat_data, bbox=bbox_small,
            transform=transform, crs="EPSG:4326", source="flat",
        )

        result = compute_slope(tile)

        # Внутрішні пікселі (де Horn gradient визначений) мають slope ≈ 0
        inner = result.degrees[1:-1, 1:-1]
        assert np.nanmean(inner) < 1.0   # < 1° для плоского

    def test_slope_range(self, synthetic_dem_tile):
        """Slope у діапазоні [0°, 90°]."""
        result = compute_slope(synthetic_dem_tile)
        valid  = result.degrees[~np.isnan(result.degrees)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 90.0)

    def test_slope_percent_vs_degrees(self, synthetic_dem_tile):
        """Slope percent = tan(degrees) × 100."""
        result = compute_slope(synthetic_dem_tile)
        inner_deg = result.degrees[1:-1, 1:-1]
        inner_pct = result.percent[1:-1, 1:-1]

        expected_pct = np.tan(np.radians(inner_deg)) * 100.0
        np.testing.assert_allclose(inner_pct, expected_pct, rtol=0.01)

    def test_slope_classify(self, synthetic_dem_tile):
        """classify() повертає uint8 масив 0..5."""
        result    = compute_slope(synthetic_dem_tile)
        classified = result.classify()

        assert classified.dtype == np.uint8
        assert np.all(classified >= 0)
        assert np.all(classified <= 5)
        assert classified.shape == result.degrees.shape

    def test_steep_slope_detection(self, bbox_small):
        """Крутий схил (45°) → slope ≈ 45°."""
        from rasterio.transform import from_bounds
        from geoengine.dem.loader import DEMTile

        # Лінійний схил: висота змінюється рівномірно
        h, w = 32, 32
        x    = np.linspace(0, 1000, w)  # 1000м по горизонталі
        data = np.tile(x, (h, 1)).astype(np.float32)  # slope = 1000/1000 = 45°

        transform = from_bounds(
            bbox_small.west, bbox_small.south,
            bbox_small.east, bbox_small.north, w, h,
        )
        tile   = DEMTile(data=data, bbox=bbox_small, transform=transform, crs="EPSG:4326", source="steep")
        result = compute_slope(tile)

        # Внутрішні пікселі мають slope > 0
        inner = result.degrees[1:-1, 1:-1]
        valid = inner[~np.isnan(inner)]
        assert len(valid) > 0
        assert np.mean(valid) > 0


# ================================================================
# ASPECT
# ================================================================

class TestAspect:

    def test_aspect_range(self, synthetic_dem_tile):
        """Aspect у діапазоні [0°, 360°]."""
        result = compute_aspect(synthetic_dem_tile)
        valid  = result.degrees[~np.isnan(result.degrees)]
        assert np.all(valid >= 0.0)
        assert np.all(valid < 360.0)

    def test_flat_terrain_has_flat_mask(self, bbox_small):
        """Плоский терейн → більшість пікселів у flat_mask."""
        from rasterio.transform import from_bounds
        from geoengine.dem.loader import DEMTile

        flat  = np.full((32, 32), 500.0, dtype=np.float32)
        t     = from_bounds(bbox_small.west, bbox_small.south, bbox_small.east, bbox_small.north, 32, 32)
        tile  = DEMTile(data=flat, bbox=bbox_small, transform=t, crs="EPSG:4326", source="flat")

        result = compute_aspect(tile)
        inner_flat = result.flat_mask[1:-1, 1:-1]
        assert np.mean(inner_flat) > 0.5   # більшість плоских

    def test_aspect_cardinal(self, synthetic_dem_tile):
        """cardinal() повертає uint8 0..7."""
        result    = compute_aspect(synthetic_dem_tile)
        cardinal  = result.cardinal()
        assert cardinal.dtype == np.uint8
        assert np.all(cardinal <= 7)


# ================================================================
# HILLSHADE
# ================================================================

class TestHillshade:

    def test_hillshade_range(self, synthetic_dem_tile):
        """Hillshade у діапазоні [0, 255]."""
        result = compute_hillshade(synthetic_dem_tile)
        valid  = result.data[~np.isnan(result.data)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 255.0)

    def test_hillshade_north_brighter_than_south(self, bbox_small):
        """При сонці з 315° (NW) → пн-зх схили яскравіші."""
        from rasterio.transform import from_bounds
        from geoengine.dem.loader import DEMTile

        # Простий схил: вище на Північ
        data = np.zeros((32, 32), dtype=np.float32)
        for i in range(32):
            data[i, :] = i * 50.0   # 0м внизу → 1550м вгорі (північ)

        t    = from_bounds(bbox_small.west, bbox_small.south, bbox_small.east, bbox_small.north, 32, 32)
        tile = DEMTile(data=data, bbox=bbox_small, transform=t, crs="EPSG:4326", source="slope_n")

        result = compute_hillshade(tile, azimuth=315, altitude=45)

        # Верхня частина (північ) яскравіша ніж нижня (південь)
        top_mean = np.nanmean(result.data[:10, :])
        bot_mean = np.nanmean(result.data[-10:, :])
        assert top_mean > bot_mean

    def test_hillshade_azimuth(self, synthetic_dem_tile):
        """Різні азимути дають різні результати."""
        h1 = compute_hillshade(synthetic_dem_tile, azimuth=0)
        h2 = compute_hillshade(synthetic_dem_tile, azimuth=180)
        assert not np.allclose(h1.data, h2.data, equal_nan=True)


# ================================================================
# CONTOURS
# ================================================================

class TestContours:

    def test_contours_count(self, synthetic_dem_tile):
        """Є хоча б декілька ізоліній."""
        result = compute_contours(synthetic_dem_tile, interval=200.0)
        assert len(result.lines) > 0
        assert len(result.elevations) == len(result.lines)

    def test_contour_elevations_multiple_of_interval(self, synthetic_dem_tile):
        """Висоти ізоліній кратні interval."""
        interval = 200.0
        result   = compute_contours(synthetic_dem_tile, interval=interval)
        for elev in result.elevations:
            assert abs(elev % interval) < 1.0   # з допуском на float

    def test_contours_with_large_interval(self, synthetic_dem_tile):
        """Великий interval → менше ізоліній."""
        r_small = compute_contours(synthetic_dem_tile, interval=100.0)
        r_large = compute_contours(synthetic_dem_tile, interval=500.0)
        assert len(r_small.lines) >= len(r_large.lines)

    def test_contour_lines_are_latlon(self, synthetic_dem_tile):
        """Координати ізоліній в розумних межах (lat/lon)."""
        result = compute_contours(synthetic_dem_tile, interval=200.0)
        for line in result.lines:
            for lat, lon in line:
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180


# ================================================================
# PROFILE
# ================================================================

class TestProfile:

    def test_profile_point_count(self, synthetic_dem_tile):
        """Профіль містить n_points точок."""
        tile   = synthetic_dem_tile
        center = tile.bbox.center
        start  = (center[0] - 0.03, center[1] - 0.03)
        end    = (center[0] + 0.03, center[1] + 0.03)

        result = compute_profile(tile, start=start, end=end, n_points=50)

        assert len(result.distances) == 50
        assert len(result.elevations) == 50
        assert len(result.lats)       == 50
        assert len(result.lons)       == 50

    def test_profile_distance_monotonic(self, synthetic_dem_tile):
        """Відстані монотонно зростають від 0."""
        tile   = synthetic_dem_tile
        center = tile.bbox.center
        start  = (center[0] - 0.02, center[1])
        end    = (center[0] + 0.02, center[1])

        result = compute_profile(tile, start=start, end=end, n_points=20)

        assert result.distances[0] == pytest.approx(0.0, abs=1.0)
        assert all(
            result.distances[i] <= result.distances[i+1]
            for i in range(len(result.distances) - 1)
        )

    def test_profile_elevations_in_range(self, synthetic_dem_tile):
        """Висоти в межах тайлу."""
        tile   = synthetic_dem_tile
        center = tile.bbox.center
        start  = (center[0] - 0.02, center[1])
        end    = (center[0] + 0.02, center[1])

        result = compute_profile(tile, start=start, end=end, n_points=20)

        valid = result.elevations[~np.isnan(result.elevations)]
        assert np.all(valid >= tile.min_elevation - 1)
        assert np.all(valid <= tile.max_elevation + 1)
