"""
GeoEngine — Analysis Service
GIS аналітика: slope, aspect, hillshade, contours, profile.
"""

from __future__ import annotations

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import structlog

from geoengine.dem.analysis import (
    compute_slope,
    compute_aspect,
    compute_hillshade,
    compute_contours,
    compute_profile,
)
from geoengine.io.geojson import contours_to_geojson

log: structlog.BoundLogger = structlog.get_logger(__name__)


class AnalysisService:
    """
    Сервіс GIS аналітики.

    Делегує CPU-bound обчислення у ThreadPool.
    Повертає wire-format dict для WS/REST.
    """

    def __init__(
        self,
        terrain_service: Any,
        max_workers: int = 2,
    ) -> None:
        self._terrain   = terrain_service
        self._executor  = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="analysis",
        )

    async def compute(
        self,
        bbox:          Any,      # BBoxModel
        analysis_type: str,
        source:        str  = "terrarium",
        options:       dict = None,
    ) -> dict[str, Any]:
        """
        Виконати аналіз.

        Args:
            bbox:          BBoxModel (pydantic)
            analysis_type: slope|aspect|hillshade|contours|profile
            source:        DEM джерело
            options:       параметри аналізу

        Returns:
            dict з результатами для WS відповіді
        """
        options = options or {}

        # Визначаємо центр bbox для вибору тайлу
        center_lat = (bbox.south + bbox.north) / 2
        center_lon = (bbox.west  + bbox.east)  / 2

        from geoengine.geo.projection import latlon_to_tile
        tile = latlon_to_tile(center_lat, center_lon, zoom=10)

        # Завантажити DEM
        dem_tile = await self._terrain._get_dem_tile(
            tile.x, tile.y, tile.z, source
        )

        t0 = time.perf_counter()

        # Виконати аналіз у ThreadPool
        result = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._run_analysis,
            dem_tile, analysis_type, options,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "analysis.done",
            type=analysis_type,
            ms=round(elapsed, 1),
        )

        return result

    @staticmethod
    def _run_analysis(
        dem_tile:      Any,
        analysis_type: str,
        options:       dict,
    ) -> dict[str, Any]:
        """
        CPU-bound аналіз. Виконується у ThreadPool.
        """
        match analysis_type:

            case "slope":
                result = compute_slope(dem_tile)
                data   = result.degrees
                return {
                    "analysis_type": "slope",
                    "result_type":   "raster",
                    "data":          _encode_f32(data),
                    "width":         data.shape[1],
                    "height":        data.shape[0],
                    "min_val":       float(np.nanmin(data)),
                    "max_val":       float(np.nanmax(data)),
                    "mean_slope_deg": round(float(result.mean_slope_deg), 2),
                }

            case "aspect":
                result = compute_aspect(dem_tile)
                data   = result.degrees
                return {
                    "analysis_type": "aspect",
                    "result_type":   "raster",
                    "data":   _encode_f32(data),
                    "width":  data.shape[1],
                    "height": data.shape[0],
                    "min_val": 0.0,
                    "max_val": 360.0,
                }

            case "hillshade":
                azimuth  = float(options.get("azimuth",  315))
                altitude = float(options.get("altitude",  45))
                z_factor = float(options.get("z_factor",   1))
                result = compute_hillshade(
                    dem_tile,
                    azimuth=azimuth,
                    altitude=altitude,
                    z_factor=z_factor,
                )
                data = result.data
                return {
                    "analysis_type": "hillshade",
                    "result_type":   "raster",
                    "data":   _encode_f32(data),
                    "width":  data.shape[1],
                    "height": data.shape[0],
                    "min_val": 0.0,
                    "max_val": 255.0,
                }

            case "contours":
                interval = float(options.get("interval", 100.0))
                base     = float(options.get("base",       0.0))
                result   = compute_contours(
                    dem_tile,
                    interval=interval,
                    base=base,
                )
                geojson = contours_to_geojson(result)
                return {
                    "analysis_type": "contours",
                    "result_type":   "vector",
                    "geojson":       geojson,
                    "line_count":    len(result.lines),
                    "interval":      interval,
                    "elevations":    result.elevations,
                }

            case "profile":
                bbox_obj = dem_tile.bbox
                c        = bbox_obj.center
                start    = options.get(
                    "start", (c[0] - 0.05, c[1] - 0.05)
                )
                end      = options.get(
                    "end",   (c[0] + 0.05, c[1] + 0.05)
                )
                n_points = int(options.get("n_points", 100))
                result   = compute_profile(
                    dem_tile,
                    start=tuple(start),
                    end=tuple(end),
                    n_points=n_points,
                )
                return {
                    "analysis_type": "profile",
                    "result_type":   "profile",
                    "profile": {
                        "distances":       result.distances.tolist(),
                        "elevations":      result.elevations.tolist(),
                        "lats":            result.lats.tolist(),
                        "lons":            result.lons.tolist(),
                        "total_length_m":  round(float(result.distances[-1]), 1),
                        "min_elevation":   round(float(np.nanmin(result.elevations)), 1),
                        "max_elevation":   round(float(np.nanmax(result.elevations)), 1),
                    }
                }

            case _:
                raise ValueError(
                    f"Невідомий тип аналізу: {analysis_type!r}. "
                    f"Доступні: slope, aspect, hillshade, contours, profile"
                )

    async def close(self) -> None:
        self._executor.shutdown(wait=False)


def _encode_f32(arr: np.ndarray) -> str:
    """Encode float32 array as base64 string."""
    clean = np.where(np.isnan(arr), 0.0, arr).astype(np.float32)
    return base64.b64encode(clean.tobytes()).decode("ascii")
