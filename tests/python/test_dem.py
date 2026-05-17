"""
GeoEngine — DEM Loader & Processor Tests
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from geoengine.dem.loader import DEMTile, DEMLoader, DEMLoadError
from geoengine.dem.processor import merge_tiles, fill_gaps, smooth
from geoengine.geo.bbox import BBox


# ================================================================
# DEMTile
# ================================================================

class TestDEMTile:

    def test_dimensions(self, synthetic_dem_tile):
        tile = synthetic_dem_tile
        assert tile.width  == 64
        assert tile.height == 64

    def test_elevation_range(self, synthetic_dem_tile):
        tile = synthetic_dem_tile
        assert tile.min_elevation >= 200.0
        assert tile.max_elevation <= 2061.0
        assert tile.min_elevation <  tile.max_elevation

    def test_mean_elevation(self, synthetic_dem_tile):
        tile = synthetic_dem_tile
        assert tile.min_elevation < tile.mean_elevation < tile.max_elevation

    def test_no_nodata_in_synthetic(self, synthetic_dem_tile):
        assert not synthetic_dem_tile.has_nodata
        assert synthetic_dem_tile.coverage_pct == pytest.approx(100.0)

    def test_has_nodata(self, dem_tile_with_nodata):
        assert dem_tile_with_nodata.has_nodata
        assert dem_tile_with_nodata.coverage_pct < 100.0
        assert dem_tile_with_nodata.coverage_pct > 0.0

    def test_sample_inside_bbox(self, synthetic_dem_tile):
        tile = synthetic_dem_tile
        lat_c, lon_c = tile.bbox.center
        elev = tile.sample(lat_c, lon_c)
        assert elev is not None
        assert tile.min_elevation <= elev <= tile.max_elevation

    def test_sample_outside_bbox(self, synthetic_dem_tile):
        tile = synthetic_dem_tile
        # Точка поза bbox
        elev = tile.sample(lat=90.0, lon=180.0)
        assert elev is None

    def test_sample_bilinear_interpolation(self, synthetic_dem_tile):
        """Білінійна інтерполяція між двома сусідніми пікселями."""
        tile = synthetic_dem_tile
        bbox = tile.bbox

        # Дві сусідні точки
        lat = (bbox.north + bbox.south) / 2
        lon1 = bbox.west + bbox.width * 0.25
        lon2 = bbox.west + bbox.width * 0.75

        elev1 = tile.sample(lat, lon1)
        elev2 = tile.sample(lat, lon2)
        # Середина між ними
        elev_mid = tile.sample(lat, (lon1 + lon2) / 2)

        assert elev1 is not None
        assert elev2 is not None
        assert elev_mid is not None
        # Середня точка між ними (з допуском на нелінійність)
        assert min(elev1, elev2) - 50 <= elev_mid <= max(elev1, elev2) + 50

    def test_resolution(self, synthetic_dem_tile):
        tile = synthetic_dem_tile
        # 64 пікселі на ~0.1° ≈ 0.0015625°/піксель
        expected_res = synthetic_dem_tile.bbox.width / 64
        assert tile.resolution_x == pytest.approx(expected_res, rel=0.01)

    def test_repr(self, synthetic_dem_tile):
        r = repr(synthetic_dem_tile)
        assert "DEMTile" in r
        assert "64×64" in r


# ================================================================
# DEMLoader
# ================================================================

class TestDEMLoader:

    def test_load_geotiff(self, tmp_geotiff, bbox_small):
        loader = DEMLoader()
        tile   = loader.load(tmp_geotiff)

        assert tile.width  > 0
        assert tile.height > 0
        assert tile.crs    == "EPSG:4326"
        assert "EPSG:4326" in tile.crs

    def test_load_with_bbox_crop(self, tmp_geotiff, bbox_small):
        loader    = DEMLoader()
        crop_bbox = BBox(
            west=bbox_small.west,
            south=bbox_small.south,
            east=bbox_small.west + bbox_small.width / 2,
            north=bbox_small.south + bbox_small.height / 2,
        )
        tile = loader.load(tmp_geotiff, bbox=crop_bbox)

        # Результат менший або рівний оригіналу
        assert tile.width  <= 64
        assert tile.height <= 64

    def test_load_nonexistent_file(self):
        loader = DEMLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/dem.tif")

    def test_nodata_replaced_with_nan(self, tmp_path, bbox_small):
        """nodata значення мають замінюватись на NaN."""
        import rasterio
        from rasterio.transform import from_bounds

        path = tmp_path / "dem_nodata.tif"
        data = np.full((32, 32), -9999.0, dtype=np.float32)
        data[5:10, 5:10] = 500.0   # Кілька валідних пікселів

        transform = from_bounds(
            bbox_small.west, bbox_small.south,
            bbox_small.east, bbox_small.north,
            32, 32,
        )

        with rasterio.open(
            path, "w", driver="GTiff",
            height=32, width=32, count=1,
            dtype=np.float32, crs="EPSG:4326",
            transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        loader = DEMLoader(fill_nodata=True)
        tile   = loader.load(path)

        assert np.isnan(tile.data[0, 0])    # nodata → NaN
        assert not np.isnan(tile.data[7, 7]) # валідний піксель

    def test_load_directory(self, tmp_path, bbox_small):
        """Завантаження всіх tif з директорії."""
        import rasterio
        from rasterio.transform import from_bounds

        # Створюємо 3 файли
        for i in range(3):
            path = tmp_path / f"dem_{i}.tif"
            data = make_synthetic_dem(16, 16, seed=i)
            t    = from_bounds(
                bbox_small.west + i * 0.001, bbox_small.south,
                bbox_small.east + i * 0.001, bbox_small.north,
                16, 16,
            )
            with rasterio.open(
                path, "w", driver="GTiff",
                height=16, width=16, count=1,
                dtype=np.float32, crs="EPSG:4326",
                transform=t,
            ) as dst:
                dst.write(data, 1)

        loader = DEMLoader()
        tiles  = loader.load_directory(tmp_path)
        assert len(tiles) == 3


def make_synthetic_dem(w, h, seed=42, min_h=200, max_h=2000):
    """Хелпер для тестів."""
    rng     = np.random.default_rng(seed)
    data    = rng.random((h, w)).astype(np.float32)
    return data * (max_h - min_h) + min_h


# ================================================================
# PROCESSOR
# ================================================================

class TestDEMProcessor:

    def test_merge_single_tile(self, synthetic_dem_tile):
        """Merge одного тайлу → той самий тайл."""
        merged = merge_tiles([synthetic_dem_tile])
        assert merged.width  == synthetic_dem_tile.width
        assert merged.height == synthetic_dem_tile.height

    def test_merge_two_tiles_horizontal(self, bbox_small):
        """Merge двох тайлів по горизонталі → ширший результат."""
        from rasterio.transform import from_bounds
        from geoengine.dem.loader import DEMTile

        # Лівий тайл
        left_bbox = BBox(
            west=bbox_small.west,
            south=bbox_small.south,
            east=(bbox_small.west + bbox_small.east) / 2,
            north=bbox_small.north,
        )
        # Правий тайл
        right_bbox = BBox(
            west=(bbox_small.west + bbox_small.east) / 2,
            south=bbox_small.south,
            east=bbox_small.east,
            north=bbox_small.north,
        )

        def make_tile(bb, seed):
            data = make_synthetic_dem(32, 32, seed=seed)
            t    = from_bounds(bb.west, bb.south, bb.east, bb.north, 32, 32)
            return DEMTile(data=data, bbox=bb, transform=t, crs="EPSG:4326", source="test")

        left  = make_tile(left_bbox,  seed=1)
        right = make_tile(right_bbox, seed=2)

        merged = merge_tiles([left, right])

        # Merged bbox охоплює обидва
        assert merged.bbox.west  <= left_bbox.west
        assert merged.bbox.east  >= right_bbox.east
        assert merged.width > left.width  # ширший ніж кожен окремо

    def test_merge_methods(self, synthetic_dem_tile):
        """Всі методи merge (first, last, mean, max, min) не кидають помилок."""
        for method in ["first", "last", "mean", "max", "min"]:
            merged = merge_tiles([synthetic_dem_tile, synthetic_dem_tile], method=method)
            assert merged.width > 0

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="порожній"):
            merge_tiles([])

    def test_fill_gaps_no_nodata(self, synthetic_dem_tile):
        """fill_gaps без NaN → повертає той самий тайл."""
        filled = fill_gaps(synthetic_dem_tile)
        assert filled is synthetic_dem_tile

    def test_fill_gaps_nearest(self, dem_tile_with_nodata):
        """fill_gaps nearest заповнює всі NaN."""
        filled = fill_gaps(dem_tile_with_nodata, method="nearest")
        assert not filled.has_nodata
        assert np.all(np.isfinite(filled.data))

    def test_fill_gaps_bilinear(self, dem_tile_with_nodata):
        """fill_gaps bilinear заповнює малі прогалини."""
        filled = fill_gaps(dem_tile_with_nodata, method="bilinear", max_gap_px=20)
        # Має заповнити 5×5 та 5×4 прогалини
        assert filled.coverage_pct > dem_tile_with_nodata.coverage_pct

    def test_smooth_gaussian(self, synthetic_dem_tile):
        """Gaussian smooth зменшує локальну варіацію."""
        original = synthetic_dem_tile
        smoothed = smooth(original, method="gaussian", sigma=2.0)

        # Середнє має бути близьким
        assert smoothed.mean_elevation == pytest.approx(
            original.mean_elevation, rel=0.05
        )
        # Але variance менша
        orig_std = float(np.nanstd(original.data))
        smth_std = float(np.nanstd(smoothed.data))
        assert smth_std < orig_std

    def test_smooth_median(self, synthetic_dem_tile):
        """Median smooth не кидає помилок."""
        smoothed = smooth(synthetic_dem_tile, method="median", sigma=1.5)
        assert smoothed.width  == synthetic_dem_tile.width
        assert smoothed.height == synthetic_dem_tile.height
