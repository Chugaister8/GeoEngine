"""
GeoEngine — DEM Analysis
Аналітичні функції поверх висотних даних:
slope, aspect, hillshade, contours, viewshed (базовий).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import structlog

from .loader import DEMTile, NODATA_DEFAULT
from ..geo.bbox import BBox
from rasterio.transform import from_bounds

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# РЕЗУЛЬТАТИ АНАЛІЗУ
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SlopeResult:
    """
    Результат аналізу крутизни схилів.
    degrees: кут нахилу у градусах (0 = горизонталь, 90 = вертикаль)
    percent:  крутизна у відсотках (tan(degrees) * 100)
    """
    degrees: npt.NDArray[np.float32]
    percent: npt.NDArray[np.float32]
    bbox:    BBox

    @property
    def mean_slope_deg(self) -> float:
        return float(np.nanmean(self.degrees))

    @property
    def max_slope_deg(self) -> float:
        return float(np.nanmax(self.degrees))

    def classify(self) -> npt.NDArray[np.uint8]:
        """
        Класифікувати схили за крутизною.

        Returns:
            uint8 масив:
              0 = плоский    (< 2°)
              1 = пологий    (2–5°)
              2 = помірний   (5–15°)
              3 = крутий     (15–30°)
              4 = дуже крутий(30–45°)
              5 = обрив      (> 45°)
        """
        d = self.degrees
        result = np.zeros_like(d, dtype=np.uint8)
        result[d >= 2]  = 1
        result[d >= 5]  = 2
        result[d >= 15] = 3
        result[d >= 30] = 4
        result[d >= 45] = 5
        return result


@dataclass(frozen=True, slots=True)
class AspectResult:
    """
    Орієнтація схилів (Aspect).
    degrees: азимут напрямку найбільшого нахилу (0=Пн, 90=Сх, 180=Пд, 270=Зх)
    flat_mask: True де схил < 1° (немає чіткої орієнтації)
    """
    degrees:   npt.NDArray[np.float32]
    flat_mask: npt.NDArray[np.bool_]
    bbox:      BBox

    def cardinal(self) -> npt.NDArray[np.uint8]:
        """
        Спрощена класифікація у 8 сторін світу.

        Returns:
            uint8: 0=Пн, 1=ПнСх, 2=Сх, 3=ПдСх, 4=Пд, 5=ПдЗх, 6=Зх, 7=ПнЗх
        """
        d = self.degrees % 360
        idx = ((d + 22.5) / 45).astype(np.uint8) % 8
        return idx


@dataclass(frozen=True, slots=True)
class HillshadeResult:
    """Результат hillshade (тіньовий рельєф)."""
    data:     npt.NDArray[np.float32]  # 0..255
    bbox:     BBox
    azimuth:  float
    altitude: float


@dataclass(frozen=True, slots=True)
class ContourResult:
    """Ізолінії рельєфу."""
    lines:    list[list[tuple[float, float]]]  # список ліній у lat/lon
    elevations: list[float]                     # висота кожної лінії
    bbox:     BBox


# ----------------------------------------------------------------
# SLOPE
# ----------------------------------------------------------------

def compute_slope(tile: DEMTile) -> SlopeResult:
    """
    Обчислити крутизну схилів (Slope).

    Алгоритм: Horn (1981) — 8-point finite difference.
    Той самий що використовує GDAL / ArcGIS / QGIS.

    Returns:
        SlopeResult з масивами degrees та percent
    """
    data     = tile.data
    cell_x_m = tile.resolution_x * 111_320.0   # градуси → метри (приблизно)
    cell_y_m = tile.resolution_y * 111_320.0

    # Горизонтальні та вертикальні градієнти (Horn 1981)
    dz_dx, dz_dy = _horn_gradient(data, cell_x_m, cell_y_m)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_pct = (np.tan(slope_rad) * 100.0).astype(np.float32)

    # Зберігаємо NaN там де вхідні дані NaN
    nan_mask = np.isnan(data)
    slope_deg[nan_mask] = np.nan
    slope_pct[nan_mask] = np.nan

    log.debug(
        "dem.slope.done",
        mean=f"{float(np.nanmean(slope_deg)):.1f}°",
        max=f"{float(np.nanmax(slope_deg)):.1f}°",
    )

    return SlopeResult(
        degrees=slope_deg,
        percent=slope_pct,
        bbox=tile.bbox,
    )


# ----------------------------------------------------------------
# ASPECT
# ----------------------------------------------------------------

def compute_aspect(tile: DEMTile) -> AspectResult:
    """
    Обчислити орієнтацію схилів (Aspect).

    Алгоритм: Horn (1981).
    Результат: азимут у градусах (0=Північ, за годинниковою).

    Returns:
        AspectResult з масивом азимутів
    """
    data     = tile.data
    cell_x_m = tile.resolution_x * 111_320.0
    cell_y_m = tile.resolution_y * 111_320.0

    dz_dx, dz_dy = _horn_gradient(data, cell_x_m, cell_y_m)

    # atan2 → азимут відносно Північ
    # dz_dy — напрямок N-S, dz_dx — напрямок E-W
    aspect_rad = np.arctan2(dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)

    # Плоскі ділянки (slope < 1°) — aspect не визначений
    slope_rad  = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    flat_mask  = slope_rad < np.radians(1.0)

    nan_mask = np.isnan(data)
    aspect_deg[nan_mask] = np.nan

    return AspectResult(
        degrees=aspect_deg,
        flat_mask=flat_mask,
        bbox=tile.bbox,
    )


# ----------------------------------------------------------------
# HILLSHADE
# ----------------------------------------------------------------

def compute_hillshade(
    tile:     DEMTile,
    azimuth:  float = 315.0,   # Пн-Зх (класичне)
    altitude: float = 45.0,    # кут сонця над горизонтом
    z_factor: float = 1.0,     # вертикальне перебільшення
) -> HillshadeResult:
    """
    Обчислити hillshade (тіньове відмивання рельєфу).

    Алгоритм: стандартний ESRI/GDAL hillshade.
    Результат — значення 0..255 (темний..світлий).

    Args:
        tile:     вхідний DEM тайл
        azimuth:  азимут сонця (0=Пн, 90=Сх, ...) у градусах
        altitude: кут підняття сонця (0=горизонт, 90=зеніт)
        z_factor: множник висоти (>1 = перебільшення рельєфу)

    Returns:
        HillshadeResult з масивом значень 0..255
    """
    data     = tile.data * z_factor
    cell_x_m = tile.resolution_x * 111_320.0
    cell_y_m = tile.resolution_y * 111_320.0

    dz_dx, dz_dy = _horn_gradient(data, cell_x_m, cell_y_m)

    # Кут нахилу поверхні
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # Азимут поверхні (aspect)
    aspect_rad = np.arctan2(dz_dx, dz_dy)

    # Кути сонця
    zenith_rad  = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)

    # Hillshade формула
    hillshade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad)
        * np.cos(azimuth_rad - aspect_rad)
    )

    # Нормалізуємо 0..255
    hillshade = np.clip(hillshade * 255.0, 0, 255).astype(np.float32)

    nan_mask = np.isnan(tile.data)
    hillshade[nan_mask] = np.nan

    return HillshadeResult(
        data=hillshade,
        bbox=tile.bbox,
        azimuth=azimuth,
        altitude=altitude,
    )


# ----------------------------------------------------------------
# CONTOURS — ізолінії
# ----------------------------------------------------------------

def compute_contours(
    tile:     DEMTile,
    interval: float = 100.0,   # метри між ізолініями
    base:     float = 0.0,     # базова висота
) -> ContourResult:
    """
    Генерувати ізолінії рельєфу.

    Алгоритм: marching squares (matplotlib).

    Args:
        tile:     вхідний DEM тайл
        interval: крок ізоліній (метри)
        base:     базова висота (метри)

    Returns:
        ContourResult зі списком ліній у lat/lon координатах
    """
    import matplotlib.pyplot as plt
    from matplotlib.contour import QuadContourSet

    data = tile.data
    valid_min = float(np.nanmin(data))
    valid_max = float(np.nanmax(data))

    # Рівні ізоліній
    first_level = np.ceil((valid_min - base) / interval) * interval + base
    levels = np.arange(first_level, valid_max + interval, interval)

    if len(levels) == 0:
        return ContourResult(lines=[], elevations=[], bbox=tile.bbox)

    # Координатні сітки
    h, w = data.shape
    cols = np.linspace(tile.bbox.west,  tile.bbox.east,  w)
    rows = np.linspace(tile.bbox.north, tile.bbox.south, h)

    # Замінюємо NaN для contour (він не підтримує NaN напряму)
    filled = np.where(np.isnan(data), valid_min - 1, data)

    fig, ax = plt.subplots()
    cs: QuadContourSet = ax.contour(cols, rows, filled, levels=levels)
    plt.close(fig)

    all_lines:      list[list[tuple[float, float]]] = []
    all_elevations: list[float] = []

    for level_idx, collection in enumerate(cs.collections):
        elev = float(levels[level_idx])
        for path in collection.get_paths():
            vertices = path.vertices  # (N, 2) array: [lon, lat]
            if len(vertices) < 2:
                continue
            line = [(float(v[1]), float(v[0])) for v in vertices]  # → (lat, lon)
            all_lines.append(line)
            all_elevations.append(elev)

    log.debug(
        "dem.contours.done",
        interval=interval,
        levels=len(levels),
        lines=len(all_lines),
    )

    return ContourResult(
        lines=all_lines,
        elevations=all_elevations,
        bbox=tile.bbox,
    )


# ----------------------------------------------------------------
# PROFILE — поперечний переріз
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Результат профілю рельєфу вздовж лінії."""
    distances: npt.NDArray[np.float64]  # відстані від початку (метри)
    elevations: npt.NDArray[np.float32] # висоти (метри)
    lats: npt.NDArray[np.float64]
    lons: npt.NDArray[np.float64]


def compute_profile(
    tile:     DEMTile,
    start:    tuple[float, float],   # (lat, lon)
    end:      tuple[float, float],   # (lat, lon)
    n_points: int = 200,
) -> ProfileResult:
    """
    Профіль рельєфу вздовж лінії між двома точками.

    Args:
        tile:     DEM тайл
        start:    початок лінії (lat, lon)
        end:      кінець лінії (lat, lon)
        n_points: кількість точок вибірки

    Returns:
        ProfileResult з відстанями та висотами
    """
    from ..geo.coords import LLH, haversine_distance

    lats = np.linspace(start[0], end[0], n_points)
    lons = np.linspace(start[1], end[1], n_points)

    elevations = np.empty(n_points, dtype=np.float32)
    for i, (lat, lon) in enumerate(zip(lats, lons, strict=True)):
        sampled = tile.sample(lat, lon)
        elevations[i] = sampled if sampled is not None else np.nan

    # Відстані від початку (метри)
    origin = LLH(lat=start[0], lon=start[1])
    distances = np.array([
        haversine_distance(origin, LLH(lat=float(lats[i]), lon=float(lons[i])))
        for i in range(n_points)
    ])

    return ProfileResult(
        distances=distances,
        elevations=elevations,
        lats=lats,
        lons=lons,
    )


# ----------------------------------------------------------------
# СПІЛЬНА МАТЕМАТИКА
# ----------------------------------------------------------------

def _horn_gradient(
    data:     npt.NDArray[np.float32],
    cell_x_m: float,
    cell_y_m: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Horn (1981) 8-point finite difference градієнт.
    Стандартний алгоритм для slope/aspect/hillshade.

    dz/dx — схід-захід градієнт
    dz/dy — північ-південь градієнт (позитивний = вгору на північ)
    """
    # Замінюємо NaN нулями для розрахунку (стандартна практика GDAL)
    d = np.where(np.isnan(data), 0.0, data).astype(np.float64)

    # Зрушення для 8 сусідів
    # [a b c]
    # [d e f]
    # [g h i]
    a = d[:-2, :-2]; b = d[:-2, 1:-1]; c = d[:-2, 2:]
    _d = d[1:-1, :-2];                  f = d[1:-1, 2:]
    g = d[2:, :-2];  h = d[2:, 1:-1];  i = d[2:, 2:]

    # Horn формула
    dz_dx = ((c + 2*f + i) - (a + 2*_d + g)) / (8.0 * cell_x_m)
    dz_dy = ((a + 2*b + c) - (g + 2*h + i)) / (8.0 * cell_y_m)

    # Результат менший на 1 піксель з кожного боку (padding)
    result_dx = np.full_like(data, np.nan)
    result_dy = np.full_like(data, np.nan)
    result_dx[1:-1, 1:-1] = dz_dx.astype(np.float32)
    result_dy[1:-1, 1:-1] = dz_dy.astype(np.float32)

    return result_dx, result_dy
