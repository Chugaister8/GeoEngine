"""
GeoEngine — Normal Map Generation
Генерація normal maps з DEM для реалістичного освітлення терейну.

Normal map кодує напрямок нормалі поверхні в кожному пікселі:
  R = X (East)
  G = Y (Up/North залежно від конвенції)
  B = Z (North або Up)

Формат: OpenGL конвенція (Y вгору, Z від глядача).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..dem.loader import DEMTile


def generate_normal_map(
    tile:     DEMTile,
    strength: float = 1.0,
) -> npt.NDArray[np.uint8]:
    """
    Генерувати normal map texture з DEM тайлу.

    Алгоритм: Sobel фільтр для градієнту → нормаль → RGB кодування.

    Args:
        tile:     DEM тайл
        strength: сила нормалі (>1 = сильніший рельєф, <1 = згладженіший)

    Returns:
        uint8 масив (H, W, 3) у RGB форматі:
          R = (nx + 1) / 2 * 255  — East компонента
          G = (ny + 1) / 2 * 255  — Up компонента
          B = (nz + 1) / 2 * 255  — North компонента
        Нейтральна нормаль (0,1,0) → (128, 255, 128)
    """
    data      = tile.data.copy()
    cell_size = tile.resolution_x * 111_320.0  # градуси → метри

    # Замінити NaN нулями для обчислення
    nan_mask = np.isnan(data)
    data[nan_mask] = 0.0

    # Sobel градієнт (більш точний ніж simple finite difference)
    from scipy.ndimage import sobel
    dz_dx = sobel(data.astype(np.float64), axis=1) / (8.0 * cell_size) * strength
    dz_dy = sobel(data.astype(np.float64), axis=0) / (8.0 * cell_size) * strength

    # Нормаль: (-dz/dx, 1, -dz/dy) нормалізована
    # (OpenGL конвенція: X=East, Y=Up, Z=North)
    nx = -dz_dx.astype(np.float32)
    ny = np.ones_like(nx)
    nz = -dz_dy.astype(np.float32)

    # Нормалізація
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    length = np.where(length == 0, 1.0, length)
    nx /= length
    ny /= length
    nz /= length

    # Кодування у RGB [0..255]
    r = ((nx + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    g = ((ny + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    b = ((nz + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

    normal_map = np.stack([r, g, b], axis=-1)

    # NaN пікселі → нейтральна нормаль (0,1,0) → (128,255,128)
    normal_map[nan_mask, 0] = 128
    normal_map[nan_mask, 1] = 255
    normal_map[nan_mask, 2] = 128

    return normal_map


def generate_normal_map_16bit(
    tile:     DEMTile,
    strength: float = 1.0,
) -> npt.NDArray[np.uint16]:
    """
    16-bit normal map для більшої точності.

    Returns:
        uint16 масив (H, W, 3).
        Нейтральна нормаль → (32768, 65535, 32768).
    """
    data      = tile.data.copy()
    cell_size = tile.resolution_x * 111_320.0
    nan_mask  = np.isnan(data)
    data[nan_mask] = 0.0

    from scipy.ndimage import sobel
    dz_dx = sobel(data.astype(np.float64), axis=1) / (8.0 * cell_size) * strength
    dz_dy = sobel(data.astype(np.float64), axis=0) / (8.0 * cell_size) * strength

    nx = -dz_dx.astype(np.float32)
    ny = np.ones_like(nx)
    nz = -dz_dy.astype(np.float32)

    length = np.sqrt(nx**2 + ny**2 + nz**2)
    length = np.where(length == 0, 1.0, length)
    nx /= length
    ny /= length
    nz /= length

    r = ((nx + 1.0) * 0.5 * 65535.0).clip(0, 65535).astype(np.uint16)
    g = ((ny + 1.0) * 0.5 * 65535.0).clip(0, 65535).astype(np.uint16)
    b = ((nz + 1.0) * 0.5 * 65535.0).clip(0, 65535).astype(np.uint16)

    normal_map = np.stack([r, g, b], axis=-1)
    normal_map[nan_mask, 0] = 32768
    normal_map[nan_mask, 1] = 65535
    normal_map[nan_mask, 2] = 32768

    return normal_map


def save_normal_map(
    normal_map: npt.NDArray[np.uint8],
    path:       str,
) -> None:
    """
    Зберегти normal map як PNG файл.

    Args:
        normal_map: uint8 масив (H, W, 3)
        path:       шлях до вихідного PNG файлу
    """
    from PIL import Image
    img = Image.fromarray(normal_map, mode="RGB")
    img.save(path, format="PNG", optimize=False)
