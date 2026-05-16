"""
GeoEngine — Analysis Service
Бізнес-логіка для GIS аналізу.

Делегує обчислення у core-python/geoengine/dem/analysis.py
та повертає wire-format dict для WebSocket/REST.
"""

from __future__ import annotations

import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
import structlog

from ..config import settings

log: structlog.BoundLogger = structlog.get_logger(__name__)

_thread_pool = ThreadPoolExecutor(
    max_workers=settings.analysis_workers,
    thread_name_prefix="geo_analysis",
)


class AnalysisService:
    """Сервіс GIS аналізу."""

    def __init__(self) -> None:
        from geoengine.dem.sources import DEMSourceManager, DEMSourceID
        self._source_mgr = DEMSourceManager(
            cache_dir=settings.dem_cache_dir,
            api_keys=settings.dem_api_keys,
        )
        self._DEMSourceID = DEMSourceID

    async def run(
        self,
        bbox_west:  float,
        bbox_south: float,
        bbox_east:  float,
        bbox_north: float,
        analyses:   list[str],
        params:     dict[str, Any],
    ) -> dict:
        """
        Виконати набір аналізів для bbox.

        Args:
            bbox_*:   географічний BBox
            analyses: список типів аналізу
            params:   параметри аналізів

        Returns:
            dict з результатами у wire-форматі
        """
        from geoengine.geo.bbox import BBox

        bbox = BBox(
            west=bbox_west, south=bbox_south,
            east=bbox_east, north=bbox_north,
        )

        # Завантажити DEM
        dem_tile = await self._source_mgr.fetch(
            bbox=bbox,
            source=self._DEMSourceID("copernicus25"),
        )

        # Виконати аналізи в thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _thread_pool,
            self._run_analyses_sync,
            dem_tile, analyses, params, bbox,
        )

        return results

    def _run_analyses_sync(
        self,
        dem_tile: Any,
        analyses: list[str],
        params:   dict[str, Any],
        bbox:     Any,
    ) -> dict:
        """Синхронне виконання аналізів у ThreadPool."""
        from geoengine.dem import analysis as ana

        all_results: dict[str, Any] = {}

        for analysis_type in analyses:
            try:
                match analysis_type:

                    case "slope":
                        result  = ana.compute_slope(dem_tile)
                        encoded = _encode_float32_raster(result.degrees)
                        all_results["slope"] = {
                            "analysis_type": "slope",
                            "bbox":          bbox.to_list(),
                            "result_type":   "raster",
                            "data":          encoded,
                            "width":         result.degrees.shape[1],
                            "height":        result.degrees.shape[0],
                            "metadata": {
                                "mean_deg":  round(result.mean_slope_deg, 2),
                                "max_deg":   round(result.max_slope_deg, 2),
                                "unit":      "degrees",
                            },
                        }

                    case "aspect":
                        result  = ana.compute_aspect(dem_tile)
                        encoded = _encode_float32_raster(result.degrees)
                        all_results["aspect"] = {
                            "analysis_type": "aspect",
                            "bbox":          bbox.to_list(),
                            "result_type":   "raster",
                            "data":          encoded,
                            "width":         result.degrees.shape[1],
                            "height":        result.degrees.shape[0],
                            "metadata":      {"unit": "degrees_from_north"},
                        }

                    case "hillshade":
                        result  = ana.compute_hillshade(
                            dem_tile,
                            azimuth=params.get("hillshade_azimuth", 315.0),
                            altitude=params.get("hillshade_altitude", 45.0),
                        )
                        encoded = _encode_float32_raster(result.data)
                        all_results["hillshade"] = {
                            "analysis_type": "hillshade",
                            "bbox":          bbox.to_list(),
                            "result_type":   "raster",
                            "data":          encoded,
                            "width":         result.data.shape[1],
                            "height":        result.data.shape[0],
                            "metadata": {
                                "azimuth":  result.azimuth,
                                "altitude": result.altitude,
                                "range":    "0-255",
                            },
                        }

                    case "contours":
                        result  = ana.compute_contours(
                            dem_tile,
                            interval=params.get("contour_interval_m", 100.0),
                        )
                        geojson = _contours_to_geojson(result)
                        all_results["contours"] = {
                            "analysis_type": "contours",
                            "bbox":          bbox.to_list(),
                            "result_type":   "geojson",
                            "data":          geojson,
                            "metadata": {
                                "interval_m": params.get("contour_interval_m", 100.0),
                                "line_count": len(result.lines),
                            },
                        }

                    case _:
                        log.warning("analysis.unknown_type", type=analysis_type)

            except Exception as exc:
                log.error("analysis.error", type=analysis_type, error=str(exc))
                all_results[analysis_type] = {"error": str(exc)}

        # Повертаємо перший результат для простоти WebSocket протоколу
        # (або можна повертати всі — залежить від клієнта)
        if all_results:
            first_key = next(iter(all_results))
            return all_results[first_key]

        return {
            "analysis_type": "none",
            "bbox":          bbox.to_list(),
            "result_type":   "value",
            "data":          "{}",
        }


def _encode_float32_raster(arr: np.ndarray) -> str:
    """Кодувати float32 масив у base64 рядок."""
    clean = np.where(np.isnan(arr), 0.0, arr).astype(np.float32)
    return base64.b64encode(clean.tobytes()).decode("ascii")


def _contours_to_geojson(result: Any) -> dict:
    """Конвертувати ContourResult у GeoJSON FeatureCollection."""
    features = []
    for line, elev in zip(result.lines, result.elevations, strict=True):
        # (lat, lon) → (lon, lat) для GeoJSON
        coords = [[lon, lat] for lat, lon in line]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "elevation": elev,
            },
        })
    return {
        "type": "FeatureCollection",
        "features": features,
    }


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """FastAPI Depends — singleton AnalysisService."""
    return AnalysisService()
