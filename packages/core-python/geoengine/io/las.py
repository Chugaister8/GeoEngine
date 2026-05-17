"""
GeoEngine — LAS/LAZ Point Cloud Reader
Читання хмар точок LiDAR у форматах LAS та LAZ.

LAS = LiDAR Archive Standard
LAZ = стиснена версія LAS (lossless)

Повертає numpy масиви для подальшої обробки:
  - points:  (N, 3) float64 — XYZ координати
  - colors:  (N, 3) uint8   — RGB (якщо є)
  - classes: (N,)   uint8   — класифікація точок
  - intensities: (N,) uint16 — інтенсивність
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)

# LAS classification codes
LAS_CLASS: Final[dict[int, str]] = {
    0:  "never_classified",
    1:  "unassigned",
    2:  "ground",
    3:  "low_vegetation",
    4:  "medium_vegetation",
    5:  "high_vegetation",
    6:  "building",
    7:  "noise",
    9:  "water",
    11: "road",
    17: "bridge",
    18: "noise_high",
}


@dataclass
class PointCloud:
    """
    Хмара точок з LAS/LAZ файлу.

    points:       (N, 3) float64 — X, Y, Z (у метрах, в CRS файлу)
    colors:       (N, 3) uint8   — RGB [0..255], або None
    intensities:  (N,)   uint16  — інтенсивність, або None
    classes:      (N,)   uint8   — LAS класифікація
    crs:          рядок EPSG або WKT
    source:       шлях до файлу
    """
    points:       npt.NDArray[np.float64]
    classes:      npt.NDArray[np.uint8]
    colors:       npt.NDArray[np.uint8] | None  = None
    intensities:  npt.NDArray[np.uint16] | None = None
    crs:          str  = "unknown"
    source:       str  = ""

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.points[:, 0]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self.points[:, 1]

    @property
    def z(self) -> npt.NDArray[np.float64]:
        return self.points[:, 2]

    @property
    def bounds(self) -> dict[str, float]:
        return {
            "x_min": float(self.x.min()),
            "x_max": float(self.x.max()),
            "y_min": float(self.y.min()),
            "y_max": float(self.y.max()),
            "z_min": float(self.z.min()),
            "z_max": float(self.z.max()),
        }

    def filter_by_class(self, *class_ids: int) -> "PointCloud":
        """Відфільтрувати точки за класом."""
        mask = np.isin(self.classes, class_ids)
        return PointCloud(
            points=self.points[mask],
            classes=self.classes[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            intensities=self.intensities[mask] if self.intensities is not None else None,
            crs=self.crs,
            source=self.source,
        )

    def ground_points(self) -> "PointCloud":
        """Тільки ground точки (клас 2)."""
        return self.filter_by_class(2)

    def building_points(self) -> "PointCloud":
        """Тільки building точки (клас 6)."""
        return self.filter_by_class(6)

    def vegetation_points(self) -> "PointCloud":
        """Vegetation точки (класи 3, 4, 5)."""
        return self.filter_by_class(3, 4, 5)

    def thin(self, voxel_size: float = 1.0) -> "PointCloud":
        """
        Voxel downsampling — зменшити кількість точок.

        Args:
            voxel_size: розмір вокселя у метрах

        Returns:
            Зменшена хмара точок (одна точка на воксель)
        """
        # Округляємо до вокселів
        voxel_idx = (self.points / voxel_size).astype(np.int64)
        # Унікальні вокселі
        _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
        unique_idx.sort()

        return PointCloud(
            points=self.points[unique_idx],
            classes=self.classes[unique_idx],
            colors=self.colors[unique_idx] if self.colors is not None else None,
            intensities=self.intensities[unique_idx] if self.intensities is not None else None,
            crs=self.crs,
            source=self.source,
        )

    def to_heightmap(
        self,
        resolution_m: float = 1.0,
    ) -> npt.NDArray[np.float32]:
        """
        Конвертувати ground точки у регулярний heightmap.

        Args:
            resolution_m: роздільна здатність (м/піксель)

        Returns:
            float32 heightmap (H, W), NaN де немає точок
        """
        from scipy.interpolate import griddata

        ground = self.ground_points()
        if ground.count == 0:
            raise ValueError("Немає ground точок для heightmap")

        b = ground.bounds
        cols = int((b["x_max"] - b["x_min"]) / resolution_m) + 1
        rows = int((b["y_max"] - b["y_min"]) / resolution_m) + 1

        xi = np.linspace(b["x_min"], b["x_max"], cols)
        yi = np.linspace(b["y_min"], b["y_max"], rows)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi = griddata(
            points=(ground.x, ground.y),
            values=ground.z,
            xi=(xi_grid, yi_grid),
            method="linear",
        )

        return zi.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"PointCloud(count={self.count:,}, "
            f"z={self.bounds['z_min']:.1f}..{self.bounds['z_max']:.1f}m, "
            f"source={Path(self.source).name})"
        )


def read_las(
    path:       str | Path,
    max_points: int | None = None,
    classes:    list[int] | None = None,
) -> PointCloud:
    """
    Прочитати LAS/LAZ файл.

    Args:
        path:       шлях до .las або .laz файлу
        max_points: максимальна кількість точок (None = всі)
        classes:    фільтр за класами LAS (None = всі)

    Returns:
        PointCloud

    Raises:
        FileNotFoundError: файл не знайдено
        ImportError:       laspy не встановлено
    """
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "Для читання LAS/LAZ потрібен laspy: pip install laspy[laszip]"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS/LAZ файл не знайдено: {path}")

    log.info("io.las.read", path=str(path))

    with laspy.open(path) as f:
        las = f.read()

    # Координати (з урахуванням offset та scale)
    x = las.x.scaled_array()
    y = las.y.scaled_array()
    z = las.z.scaled_array()

    points  = np.stack([x, y, z], axis=-1).astype(np.float64)
    classes_arr = np.array(las.classification, dtype=np.uint8)

    # Фільтр за класами
    if classes is not None:
        mask    = np.isin(classes_arr, classes)
        points  = points[mask]
        classes_arr = classes_arr[mask]
    else:
        mask = None

    # Downsampling якщо потрібно
    if max_points is not None and len(points) > max_points:
        step   = len(points) // max_points
        points = points[::step]
        classes_arr = classes_arr[::step]
        log.info(
            "io.las.downsample",
            original=len(points) * step,
            kept=len(points),
        )

    # Кольори (RGB якщо є)
    colors: npt.NDArray[np.uint8] | None = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        try:
            r = np.array(las.red,   dtype=np.uint16)
            g = np.array(las.green, dtype=np.uint16)
            b = np.array(las.blue,  dtype=np.uint16)

            if mask is not None:
                r, g, b = r[mask], g[mask], b[mask]
            if max_points is not None and len(r) > max_points:
                step = len(r) // max_points
                r, g, b = r[::step], g[::step], b[::step]

            # LAS зберігає 16-bit, конвертуємо до 8-bit
            r8 = (r >> 8).astype(np.uint8)
            g8 = (g >> 8).astype(np.uint8)
            b8 = (b >> 8).astype(np.uint8)
            colors = np.stack([r8, g8, b8], axis=-1)
        except Exception:
            pass  # Кольори не обов'язкові

    # Інтенсивність
    intensities: npt.NDArray[np.uint16] | None = None
    if hasattr(las, "intensity"):
        try:
            inten = np.array(las.intensity, dtype=np.uint16)
            if mask is not None:
                inten = inten[mask]
            if max_points is not None and len(inten) > max_points:
                inten = inten[::step]
            intensities = inten
        except Exception:
            pass

    # CRS
    crs_str = "unknown"
    if hasattr(las, "header") and hasattr(las.header, "parse_crs"):
        try:
            crs_str = str(las.header.parse_crs())
        except Exception:
            pass

    pc = PointCloud(
        points=points,
        classes=classes_arr,
        colors=colors,
        intensities=intensities,
        crs=crs_str,
        source=str(path),
    )

    log.info(
        "io.las.done",
        count=f"{pc.count:,}",
        z=f"{pc.bounds['z_min']:.1f}..{pc.bounds['z_max']:.1f}m",
    )
    return pc
