"""
GeoEngine — DEM Loader
Завантаження висотних даних з локальних файлів.
Підтримує: GeoTIFF, HGT (SRTM), ASC (ASCII Grid).

Всі дані нормалізуються до спільного формату DEMTile —
float32 numpy масив висот у метрах + метадані.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.mask
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

import structlog

from ..geo.bbox import BBox
from ..geo.projection import TileXYZ, tile_to_bbox

# ----------------------------------------------------------------
# Логер
# ----------------------------------------------------------------

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------------

NODATA_DEFAULT: Final[float] = -9999.0
TARGET_CRS:     Final[str]   = "EPSG:4326"   # WGS84 — наш стандарт
FLOAT32_MAX:    Final[float] = np.finfo(np.float32).max


# ----------------------------------------------------------------
# ТИПИ ДАНИХ
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DEMTile:
    """
    Один тайл висотних даних — результат будь-якого завантаження.

    data:      float32 масив форми (height, width) — висоти в метрах.
               NaN де дані відсутні.
    bbox:      географічний BBox тайлу (WGS84)
    transform: Affine трансформація піксель → координати
    crs:       система координат (зазвичай EPSG:4326)
    source:    звідки завантажено (шлях або URL)
    nodata:    оригінальне значення "немає даних" до заміни на NaN
    """
    data:      npt.NDArray[np.float32]
    bbox:      BBox
    transform: Affine
    crs:       str
    source:    str
    nodata:    float = NODATA_DEFAULT

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def resolution_x(self) -> float:
        """Роздільна здатність по X (градуси або метри залежно від CRS)."""
        return float(abs(self.transform.a))

    @property
    def resolution_y(self) -> float:
        """Роздільна здатність по Y."""
        return float(abs(self.transform.e))

    @property
    def min_elevation(self) -> float:
        """Мінімальна висота (ігноруючи NaN)."""
        valid = self.data[~np.isnan(self.data)]
        return float(np.min(valid)) if len(valid) > 0 else 0.0

    @property
    def max_elevation(self) -> float:
        """Максимальна висота (ігноруючи NaN)."""
        valid = self.data[~np.isnan(self.data)]
        return float(np.max(valid)) if len(valid) > 0 else 0.0

    @property
    def mean_elevation(self) -> float:
        """Середня висота."""
        return float(np.nanmean(self.data))

    @property
    def has_nodata(self) -> bool:
        """Чи є відсутні дані в тайлі."""
        return bool(np.any(np.isnan(self.data)))

    @property
    def coverage_pct(self) -> float:
        """Відсоток пікселів з валідними даними (0..100)."""
        total = self.data.size
        if total == 0:
            return 0.0
        valid = int(np.sum(~np.isnan(self.data)))
        return valid / total * 100.0

    def sample(self, lat: float, lon: float) -> float | None:
        """
        Біліарна інтерполяція висоти в точці (lat, lon).

        Returns:
            Висота в метрах або None якщо точка поза тайлом або nodata.
        """
        if (lat, lon) not in self.bbox:
            return None

        # Перетворити lat/lon → піксельні координати
        col = (lon - self.bbox.west)  / self.resolution_x
        row = (self.bbox.north - lat) / self.resolution_y

        # Біліарна інтерполяція
        col0, row0 = int(col), int(row)
        col1, row1 = min(col0 + 1, self.width - 1), min(row0 + 1, self.height - 1)

        dc = col - col0
        dr = row - row0

        h00 = self.data[row0, col0]
        h10 = self.data[row0, col1]
        h01 = self.data[row1, col0]
        h11 = self.data[row1, col1]

        # Якщо будь-який з 4 пікселів NaN — повертаємо None
        if any(np.isnan(h) for h in [h00, h10, h01, h11]):
            return None

        value = (h00 * (1 - dc) * (1 - dr)
                 + h10 * dc       * (1 - dr)
                 + h01 * (1 - dc) * dr
                 + h11 * dc       * dr)
        return float(value)

    def __repr__(self) -> str:
        return (
            f"DEMTile({self.width}×{self.height}px, "
            f"elev={self.min_elevation:.0f}..{self.max_elevation:.0f}m, "
            f"coverage={self.coverage_pct:.1f}%, "
            f"src={Path(self.source).name})"
        )


# ----------------------------------------------------------------
# LOADER
# ----------------------------------------------------------------

class DEMLoader:
    """
    Завантажувач DEM файлів.

    Підтримує будь-який формат який читає rasterio:
    GeoTIFF, HGT (SRTM), ASC, BIL, IMG, ...

    Автоматично:
    - Репроєктує у WGS84 якщо потрібно
    - Замінює nodata → NaN
    - Обрізає до bbox якщо вказано
    - Ресемплює до потрібної роздільної здатності

    Usage:
        loader = DEMLoader()
        tile = loader.load("path/to/dem.tif")
        tile = loader.load("dem.tif", bbox=BBox(22, 47, 24, 49))
    """

    def __init__(
        self,
        reproject_to_wgs84: bool = True,
        fill_nodata:        bool = True,
    ) -> None:
        """
        Args:
            reproject_to_wgs84: автоматично репроєктувати у EPSG:4326
            fill_nodata:        замінювати nodata на NaN
        """
        self._reproject    = reproject_to_wgs84
        self._fill_nodata  = fill_nodata

    # ---- Публічний API ----

    def load(
        self,
        path:        str | Path,
        bbox:        BBox | None = None,
        target_res:  float | None = None,  # метри/піксель, None = оригінал
    ) -> DEMTile:
        """
        Завантажити DEM файл.

        Args:
            path:       шлях до файлу (GeoTIFF, HGT, ASC, ...)
            bbox:       обрізати до цього BBox (необов'язково)
            target_res: ресемпл до цієї роздільної здатності (м/піксель)

        Returns:
            DEMTile з даними висот

        Raises:
            FileNotFoundError: файл не існує
            DEMLoadError:      помилка читання або неvalid формат
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DEM файл не знайдено: {path}")

        log.info("dem.load.start", path=str(path), bbox=str(bbox))

        try:
            with rasterio.open(path) as src:
                dem_tile = self._read_rasterio(
                    src=src,
                    source_str=str(path),
                    bbox=bbox,
                    target_res=target_res,
                )
        except rasterio.errors.RasterioIOError as exc:
            raise DEMLoadError(f"Не вдалося відкрити {path}: {exc}") from exc

        log.info(
            "dem.load.done",
            path=str(path),
            size=f"{dem_tile.width}×{dem_tile.height}",
            elev=f"{dem_tile.min_elevation:.0f}..{dem_tile.max_elevation:.0f}m",
        )
        return dem_tile

    def load_for_tile(
        self,
        path: str | Path,
        tile: TileXYZ,
    ) -> DEMTile:
        """
        Завантажити DEM для конкретного XYZ тайлу.

        Args:
            path: шлях до DEM файлу
            tile: XYZ адреса тайлу

        Returns:
            DEMTile обрізаний до bbox тайлу
        """
        bbox = tile_to_bbox(tile)
        return self.load(path=path, bbox=bbox)

    def load_directory(
        self,
        directory:  str | Path,
        bbox:       BBox | None = None,
        pattern:    str = "**/*.tif",
    ) -> list[DEMTile]:
        """
        Завантажити всі DEM файли з директорії.

        Args:
            directory: шлях до директорії
            bbox:      фільтр по bbox (завантажуємо тільки ті що перетинаються)
            pattern:   glob паттерн для пошуку файлів

        Returns:
            Список DEMTile, впорядкований за іменем файлу
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Не директорія: {directory}")

        files = sorted(directory.glob(pattern))
        if not files:
            log.warning("dem.load_dir.empty", directory=str(directory), pattern=pattern)
            return []

        tiles: list[DEMTile] = []
        for f in files:
            try:
                # Спочатку перевіримо bbox файлу без повного завантаження
                if bbox is not None and not self._file_intersects_bbox(f, bbox):
                    log.debug("dem.load_dir.skip", file=f.name, reason="bbox_mismatch")
                    continue
                tile = self.load(path=f, bbox=bbox)
                tiles.append(tile)
            except (DEMLoadError, FileNotFoundError) as exc:
                log.warning("dem.load_dir.error", file=str(f), error=str(exc))
                continue

        log.info("dem.load_dir.done", count=len(tiles), directory=str(directory))
        return tiles

    # ---- Приватні методи ----

    def _read_rasterio(
        self,
        src:        rasterio.DatasetReader,
        source_str: str,
        bbox:       BBox | None,
        target_res: float | None,
    ) -> DEMTile:
        """Основна логіка читання через rasterio."""

        # 1. Репроєкція якщо потрібно
        if self._reproject and src.crs != CRS.from_epsg(4326):
            return self._reproject_and_read(src, source_str, bbox, target_res)

        # 2. Обрізання до bbox
        if bbox is not None:
            data, transform = self._crop_to_bbox(src, bbox)
        else:
            data      = src.read(1).astype(np.float32)
            transform = src.transform

        # 3. Обробка nodata
        nodata = src.nodata if src.nodata is not None else NODATA_DEFAULT
        if self._fill_nodata:
            data = self._replace_nodata(data, nodata)

        # 4. Побудова BBox
        actual_bbox = self._transform_to_bbox(transform, data.shape, src.crs)

        # 5. Ресемпл якщо потрібно
        if target_res is not None:
            data, transform = self._resample(
                data, transform, src.crs, target_res
            )

        return DEMTile(
            data=data,
            bbox=actual_bbox,
            transform=transform,
            crs=str(src.crs),
            source=source_str,
            nodata=float(nodata) if nodata is not None else NODATA_DEFAULT,
        )

    def _crop_to_bbox(
        self,
        src:  rasterio.DatasetReader,
        bbox: BBox,
    ) -> tuple[npt.NDArray[np.float32], Affine]:
        """Обрізати растр до bbox."""
        from shapely.geometry import box as shapely_box

        geom = shapely_box(bbox.west, bbox.south, bbox.east, bbox.north)

        try:
            data, transform = rasterio.mask.mask(
                src,
                [geom.__geo_interface__],
                crop=True,
                filled=True,
                nodata=src.nodata if src.nodata is not None else NODATA_DEFAULT,
            )
            return data[0].astype(np.float32), transform
        except ValueError as exc:
            # bbox не перетинається з растром
            raise DEMLoadError(
                f"BBox {bbox} не перетинається з файлом"
            ) from exc

    def _reproject_and_read(
        self,
        src:        rasterio.DatasetReader,
        source_str: str,
        bbox:       BBox | None,
        target_res: float | None,
    ) -> DEMTile:
        """Репроєктувати у WGS84 і прочитати."""
        import rasterio.warp

        dst_crs = CRS.from_epsg(4326)

        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        data = np.empty((height, width), dtype=np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA_DEFAULT

        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

        if self._fill_nodata:
            data = self._replace_nodata(data, float(nodata))

        actual_bbox = self._transform_to_bbox(transform, data.shape, dst_crs)

        # Тепер обрізаємо якщо bbox заданий
        if bbox is not None:
            data, actual_bbox, transform = self._crop_array(
                data, actual_bbox, transform, bbox
            )

        return DEMTile(
            data=data,
            bbox=actual_bbox,
            transform=transform,
            crs=str(dst_crs),
            source=source_str,
            nodata=float(nodata),
        )

    def _crop_array(
        self,
        data:      npt.NDArray[np.float32],
        src_bbox:  BBox,
        transform: Affine,
        crop_bbox: BBox,
    ) -> tuple[npt.NDArray[np.float32], BBox, Affine]:
        """Обрізати numpy масив до bbox."""
        h, w = data.shape
        res_x = abs(transform.a)
        res_y = abs(transform.e)

        # Піксельні межі обрізання
        col0 = max(0, int((crop_bbox.west  - src_bbox.west)  / res_x))
        col1 = min(w, int((crop_bbox.east  - src_bbox.west)  / res_x) + 1)
        row0 = max(0, int((src_bbox.north  - crop_bbox.north) / res_y))
        row1 = min(h, int((src_bbox.north  - crop_bbox.south) / res_y) + 1)

        cropped = data[row0:row1, col0:col1]

        new_west  = src_bbox.west  + col0 * res_x
        new_north = src_bbox.north - row0 * res_y
        new_east  = new_west  + cropped.shape[1] * res_x
        new_south = new_north - cropped.shape[0] * res_y

        new_bbox = BBox(
            west=new_west, south=new_south,
            east=new_east, north=new_north,
        )
        new_transform = Affine(
            transform.a, 0, new_west,
            0, transform.e, new_north,
        )
        return cropped, new_bbox, new_transform

    @staticmethod
    def _replace_nodata(
        data:   npt.NDArray[np.float32],
        nodata: float,
    ) -> npt.NDArray[np.float32]:
        """Замінити nodata значення на NaN."""
        result = data.copy()
        # Враховуємо float точність
        if np.isnan(nodata):
            return result  # вже NaN
        mask = np.abs(result - nodata) < 1.0  # ±1м допуск для float32
        # Також виключаємо аномально великі/малі значення
        mask |= result > 9000.0    # вище Евересту — явно nodata
        mask |= result < -500.0    # нижче Мертвого моря — явно nodata
        result[mask] = np.nan
        return result

    @staticmethod
    def _transform_to_bbox(
        transform: Affine,
        shape:     tuple[int, ...],
        crs:       CRS,
    ) -> BBox:
        """Отримати BBox з Affine трансформації та розміру растру."""
        h, w = shape[0], shape[1]
        west  = transform.c
        north = transform.f
        east  = west  + w * transform.a
        south = north + h * transform.e  # transform.e від'ємний

        # Якщо CRS не WGS84 — просто повертаємо як є
        # (репроєкція відбулась раніше)
        return BBox(
            west=min(west, east),
            south=min(south, north),
            east=max(west, east),
            north=max(south, north),
        )

    @staticmethod
    def _resample(
        data:       npt.NDArray[np.float32],
        transform:  Affine,
        crs:        CRS,
        target_m:   float,
    ) -> tuple[npt.NDArray[np.float32], Affine]:
        """
        Ресемпл масиву до target_m метрів/піксель.
        Використовує bilinear інтерполяцію.
        """
        import rasterio.warp

        current_res = abs(transform.a)  # градуси
        # Приблизне перетворення метри → градуси (на екваторі 1° ≈ 111320м)
        target_deg = target_m / 111_320.0

        if abs(current_res - target_deg) / target_deg < 0.05:
            return data, transform  # вже близько — не ресемплюємо

        scale = current_res / target_deg
        new_h = max(1, int(data.shape[0] * scale))
        new_w = max(1, int(data.shape[1] * scale))

        new_transform = Affine(
            target_deg, 0, transform.c,
            0, -target_deg, transform.f,
        )

        resampled = np.empty((new_h, new_w), dtype=np.float32)
        rasterio.warp.reproject(
            source=data,
            destination=resampled,
            src_transform=transform,
            src_crs=crs,
            dst_transform=new_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
        return resampled, new_transform

    @staticmethod
    def _file_intersects_bbox(path: Path, bbox: BBox) -> bool:
        """Перевірити bbox файлу без повного завантаження."""
        try:
            with rasterio.open(path) as src:
                b = src.bounds
                file_bbox = BBox(
                    west=b.left, south=b.bottom,
                    east=b.right, north=b.top,
                )
                return file_bbox.intersects(bbox)
        except Exception:
            return True  # якщо не вдалось перевірити — завантажуємо


# ----------------------------------------------------------------
# EXCEPTIONS
# ----------------------------------------------------------------

class DEMLoadError(Exception):
    """Помилка завантаження DEM даних."""
    pass
