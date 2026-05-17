"""
GeoEngine — GeoTIFF I/O
Читання та запис GeoTIFF файлів.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds

from ..geo.bbox import BBox
from ..dem.loader import DEMTile

log: structlog.BoundLogger = structlog.get_logger(__name__)


def write_geotiff(
    path:      str | Path,
    data:      npt.NDArray[np.float32],
    bbox:      BBox,
    crs:       str = "EPSG:4326",
    nodata:    float = -9999.0,
    compress:  Literal["lzw", "deflate", "none"] = "lzw",
    overwrite: bool = False,
) -> Path:
    """
    Записати numpy масив як GeoTIFF.

    Args:
        path:      вихідний файл
        data:      float32 масив (H, W)
        bbox:      географічний BBox
        crs:       система координат
        nodata:    значення nodata
        compress:  стиснення
        overwrite: перезаписати якщо існує

    Returns:
        Path до записаного файлу

    Raises:
        FileExistsError: якщо файл існує і overwrite=False
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Файл вже існує: {path}. Використай overwrite=True.")

    path.parent.mkdir(parents=True, exist_ok=True)

    h, w    = data.shape
    transform = from_bounds(bbox.west, bbox.south, bbox.east, bbox.north, w, h)

    # Замінити NaN → nodata перед записом
    data_out = np.where(np.isnan(data), nodata, data).astype(np.float32)

    compress_opt = {} if compress == "none" else {"compress": compress}

    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=np.float32,
        crs=CRS.from_string(crs),
        transform=transform,
        nodata=nodata,
        **compress_opt,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data_out, 1)

    log.info(
        "io.geotiff.write",
        path=str(path),
        size=f"{w}×{h}",
        bbox=str(bbox),
        compress=compress,
    )
    return path


def dem_tile_to_geotiff(
    tile:      DEMTile,
    path:      str | Path,
    **kwargs,
) -> Path:
    """Зберегти DEMTile як GeoTIFF."""
    return write_geotiff(
        path=path,
        data=tile.data,
        bbox=tile.bbox,
        crs=tile.crs,
        nodata=tile.nodata,
        **kwargs,
    )


def read_geotiff_meta(path: str | Path) -> dict:
    """
    Прочитати метадані GeoTIFF без завантаження даних.
    Швидко для великих файлів.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        return {
            "path":       str(path),
            "width":      src.width,
            "height":     src.height,
            "crs":        str(src.crs),
            "transform":  list(src.transform)[:6],
            "nodata":     src.nodata,
            "dtype":      str(src.dtypes[0]),
            "bounds": {
                "west":  src.bounds.left,
                "south": src.bounds.bottom,
                "east":  src.bounds.right,
                "north": src.bounds.top,
            },
        }
