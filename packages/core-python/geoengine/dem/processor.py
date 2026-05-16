"""
GeoEngine — DEM Processor
Обробка та трансформація висотних даних:
нормалізація, merging, заповнення прогалин, фільтрація.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import structlog
from rasterio.transform import from_bounds

from ..geo.bbox import BBox
from .loader import DEMTile, NODATA_DEFAULT

log: structlog.BoundLogger = structlog.get_logger(__name__)


# ----------------------------------------------------------------
# MERGE — злиття кількох тайлів в один
# ----------------------------------------------------------------

def merge_tiles(
    tiles:      list[DEMTile],
    target_res: float | None = None,
    method:     Literal["first", "last", "mean", "max", "min"] = "first",
) -> DEMTile:
    """
    Злити список DEMTile в один суцільний тайл.

    Алгоритм:
    1. Обчислити загальний BBox
    2. Створити порожній output масив (NaN)
    3. Вставити кожен тайл у правильну позицію
    4. Конфлікти вирішувати за method

    Args:
        tiles:      список тайлів для злиття (мають перекриватись або стикатись)
        target_res: цільова роздільна здатність (None = взяти з першого тайлу)
        method:     як обробляти overlap між тайлами

    Returns:
        Один DEMTile що охоплює всі вхідні тайли

    Raises:
        ValueError: якщо tiles порожній
    """
    if not tiles:
        raise ValueError("Список тайлів для злиття порожній")
    if len(tiles) == 1:
        return tiles[0]

    # Загальний BBox
    total_bbox = tiles[0].bbox
    for t in tiles[1:]:
        total_bbox = total_bbox.union(t.bbox)

    # Роздільна здатність
    res = target_res
    if res is None:
        # Беремо найдрібнішу (найменшу) роздільну здатність
        res = min(t.resolution_x for t in tiles)

    # Розміри вихідного масиву
    out_w = max(1, int(round(total_bbox.width  / res)))
    out_h = max(1, int(round(total_bbox.height / res)))

    output     = np.full((out_h, out_w), np.nan, dtype=np.float32)
    count      = np.zeros((out_h, out_w), dtype=np.int32)

    for tile in tiles:
        _paste_tile(output, count, tile, total_bbox, res, method)

    # Для method="mean" — ділимо на кількість
    if method == "mean":
        valid = count > 0
        output[valid] /= count[valid].astype(np.float32)

    transform = from_bounds(
        total_bbox.west, total_bbox.south,
        total_bbox.east, total_bbox.north,
        out_w, out_h,
    )

    log.info(
        "dem.merge.done",
        tiles=len(tiles),
        size=f"{out_w}×{out_h}",
        bbox=str(total_bbox),
    )

    return DEMTile(
        data=output,
        bbox=total_bbox,
        transform=transform,
        crs=tiles[0].crs,
        source=f"merged({len(tiles)} tiles)",
        nodata=NODATA_DEFAULT,
    )


def _paste_tile(
    output:    npt.NDArray[np.float32],
    count:     npt.NDArray[np.int32],
    tile:      DEMTile,
    out_bbox:  BBox,
    res:       float,
    method:    str,
) -> None:
    """Вставити один тайл у вихідний масив."""
    out_h, out_w = output.shape

    # Позиція тайлу в output масиві
    col_start = int(round((tile.bbox.west  - out_bbox.west)  / res))
    row_start = int(round((out_bbox.north  - tile.bbox.north) / res))

    # Ресемпл тайлу якщо потрібно
    src_data = tile.data
    if abs(tile.resolution_x - res) / res > 0.01:
        src_data = _resample_array(src_data, tile.resolution_x, res)

    src_h, src_w = src_data.shape

    # Межі вставки (обрізаємо якщо виходить за output)
    col_end = min(out_w, col_start + src_w)
    row_end = min(out_h, row_start + src_h)
    col_start_clip = max(0, col_start)
    row_start_clip = max(0, row_start)

    src_col0 = col_start_clip - col_start
    src_row0 = row_start_clip - row_start
    src_col1 = src_col0 + (col_end - col_start_clip)
    src_row1 = src_row0 + (row_end - row_start_clip)

    if src_col1 <= src_col0 or src_row1 <= src_row0:
        return  # тайл поза output

    src_patch = src_data[src_row0:src_row1, src_col0:src_col1]
    out_patch = output[row_start_clip:row_end, col_start_clip:col_end]
    cnt_patch = count[row_start_clip:row_end,  col_start_clip:col_end]

    valid_src = ~np.isnan(src_patch)

    if method == "first":
        # Заповнюємо тільки NaN пікселі
        mask = valid_src & np.isnan(out_patch)
        out_patch[mask] = src_patch[mask]

    elif method == "last":
        # Перезаписуємо якщо є дані
        out_patch[valid_src] = src_patch[valid_src]

    elif method == "mean":
        # Накопичуємо для пізнішого ділення
        out_patch[valid_src] = np.where(
            np.isnan(out_patch[valid_src]),
            src_patch[valid_src],
            out_patch[valid_src] + src_patch[valid_src],
        )
        cnt_patch[valid_src] += 1

    elif method == "max":
        mask = valid_src & (~np.isnan(out_patch))
        out_patch[mask] = np.maximum(out_patch[mask], src_patch[mask])
        # Де output був NaN — просто вставляємо
        new_mask = valid_src & np.isnan(out_patch)
        out_patch[new_mask] = src_patch[new_mask]

    elif method == "min":
        mask = valid_src & (~np.isnan(out_patch))
        out_patch[mask] = np.minimum(out_patch[mask], src_patch[mask])
        new_mask = valid_src & np.isnan(out_patch)
        out_patch[new_mask] = src_patch[new_mask]


def _resample_array(
    data:     npt.NDArray[np.float32],
    src_res:  float,
    dst_res:  float,
) -> npt.NDArray[np.float32]:
    """Простий ресемпл через scipy zoom."""
    from scipy.ndimage import zoom as scipy_zoom
    scale = src_res / dst_res
    return scipy_zoom(data, scale, order=1, mode="nearest").astype(np.float32)


# ----------------------------------------------------------------
# FILL GAPS — заповнення прогалин
# ----------------------------------------------------------------

def fill_gaps(
    tile:   DEMTile,
    method: Literal["bilinear", "nearest", "kriging"] = "bilinear",
    max_gap_px: int = 50,
) -> DEMTile:
    """
    Заповнити прогалини (NaN) у DEM даних.

    Args:
        tile:       вхідний тайл з NaN прогалинами
        method:     метод інтерполяції
        max_gap_px: максимальний розмір прогалини в пікселях для заповнення

    Returns:
        Новий DEMTile з заповненими прогалинами
    """
    if not tile.has_nodata:
        return tile  # нема NaN — нема роботи

    data   = tile.data.copy()
    nan_mask = np.isnan(data)
    nan_count = int(np.sum(nan_mask))

    log.debug("dem.fill_gaps.start", method=method, nan_pixels=nan_count)

    if method == "nearest":
        filled = _fill_nearest(data, nan_mask)
    elif method == "bilinear":
        filled = _fill_bilinear(data, nan_mask, max_gap_px)
    else:
        raise NotImplementedError(f"Метод '{method}' ще не реалізований")

    log.debug(
        "dem.fill_gaps.done",
        filled=int(np.sum(np.isnan(filled) < nan_mask)),
    )

    return DEMTile(
        data=filled,
        bbox=tile.bbox,
        transform=tile.transform,
        crs=tile.crs,
        source=f"{tile.source}[filled:{method}]",
        nodata=tile.nodata,
    )


def _fill_nearest(
    data:     npt.NDArray[np.float32],
    nan_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    """Заповнення методом найближчого сусіда (scipy)."""
    from scipy.ndimage import distance_transform_edt

    _, indices = distance_transform_edt(
        nan_mask,
        return_distances=True,
        return_indices=True,
    )
    filled = data[tuple(indices)]
    return filled.astype(np.float32)


def _fill_bilinear(
    data:       npt.NDArray[np.float32],
    nan_mask:   npt.NDArray[np.bool_],
    max_gap_px: int,
) -> npt.NDArray[np.float32]:
    """
    Заповнення через griddata bilinear інтерполяцію.
    Заповнює тільки прогалини <= max_gap_px.
    """
    from scipy.ndimage import label, binary_dilation
    from scipy.interpolate import griddata

    result = data.copy()

    # Знаходимо зв'язані компоненти NaN регіонів
    labeled, num_features = label(nan_mask)

    rows, cols = np.indices(data.shape)
    valid_mask = ~nan_mask

    # Координати та значення валідних точок
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    valid_vals = data[valid_mask]

    for region_id in range(1, num_features + 1):
        region = labeled == region_id
        region_size = int(np.sum(region))

        if region_size > max_gap_px * max_gap_px:
            continue  # пропускаємо великі прогалини

        # Точки для інтерполяції + невелика зона навколо
        context = binary_dilation(region, iterations=3)
        context_valid = context & valid_mask

        if int(np.sum(context_valid)) < 4:
            continue  # недостатньо точок

        src_rows = rows[context_valid]
        src_cols = cols[context_valid]
        src_vals = data[context_valid]

        tgt_rows = rows[region]
        tgt_cols = cols[region]

        try:
            interpolated = griddata(
                points=(src_rows, src_cols),
                values=src_vals,
                xi=(tgt_rows, tgt_cols),
                method="linear",
            )
            result[region] = interpolated.astype(np.float32)
        except Exception:
            pass  # якщо не вийшло — залишаємо NaN

    return result


# ----------------------------------------------------------------
# SMOOTH — згладжування
# ----------------------------------------------------------------

def smooth(
    tile:   DEMTile,
    method: Literal["gaussian", "median"] = "gaussian",
    sigma:  float = 1.0,
) -> DEMTile:
    """
    Згладити DEM дані.

    Корисно для видалення артефактів після інтерполяції
    або зменшення шуму в LiDAR даних.

    Args:
        tile:   вхідний тайл
        method: gaussian (швидший) або median (зберігає краї)
        sigma:  параметр згладжування (більше = сильніше)

    Returns:
        Новий згладжений DEMTile
    """
    from scipy.ndimage import gaussian_filter, median_filter

    data = tile.data.copy()

    # Тимчасово замінюємо NaN середнім для фільтрації
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        mean_val = float(np.nanmean(data))
        data[nan_mask] = mean_val

    if method == "gaussian":
        smoothed = gaussian_filter(data, sigma=sigma).astype(np.float32)
    elif method == "median":
        kernel_size = max(3, int(sigma * 2 + 1)) | 1  # непарне число
        smoothed = median_filter(data, size=kernel_size).astype(np.float32)
    else:
        raise NotImplementedError(f"Метод '{method}' не підтримується")

    # Відновлюємо NaN
    smoothed[nan_mask] = np.nan

    return DEMTile(
        data=smoothed,
        bbox=tile.bbox,
        transform=tile.transform,
        crs=tile.crs,
        source=f"{tile.source}[smooth:{method}:{sigma}]",
        nodata=tile.nodata,
  )
