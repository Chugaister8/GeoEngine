"""
GeoEngine — Seed Data Script
Завантажує реальні DEM тайли Карпат без API ключа
(Terrarium tiles — публічний AWS S3, безкоштовно).

Запуск:
    python scripts/seed_data.py

Результат:
    data/tiles/terrarium/{z}/{x}/{y}.png  — PNG тайли
    data/dem/carpathians.tif              — зведений GeoTIFF
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

# ── Карпати bbox ────────────────────────────────────────────────
BBOX   = (23.0, 47.8, 25.5, 49.2)   # west, south, east, north
ZOOM   = 9                            # zoom 9 ≈ 300м/піксель (швидко)
OUTDIR = Path("data")

TERRARIUM_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
    "/{z}/{x}/{y}.png"
)


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    import math
    n   = 1 << zoom
    x   = int((lon + 180) / 360 * n)
    lat_r = math.radians(lat)
    y   = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return max(0, min(n-1, x)), max(0, min(n-1, y))


def tile_to_bbox(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    import math
    n     = 1 << z
    west  = x / n * 360 - 180
    east  = (x+1) / n * 360 - 180
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*(y+1)/n))))
    return west, south, east, north


def decode_terrarium(img: Image.Image) -> np.ndarray:
    """RGB PNG → float32 висоти в метрах."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    elev = (R * 256.0 + G + B / 256.0) - 32768.0
    # Нульові пікселі (море/nodata)
    nodata = (R == 0) & (G == 0) & (B == 0)
    elev[nodata] = np.nan
    return elev


async def download_tile(
    client: httpx.AsyncClient,
    x: int, y: int, z: int,
    out_dir: Path,
) -> tuple[int, int, np.ndarray] | None:
    cache = out_dir / "terrarium" / str(z) / str(x) / f"{y}.png"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        url = TERRARIUM_URL.format(z=z, x=x, y=y)
        try:
            r = await client.get(url, timeout=30)
            if r.status_code == 200:
                cache.write_bytes(r.content)
            else:
                print(f"  ⚠ HTTP {r.status_code}: {x}/{y}")
                return None
        except Exception as e:
            print(f"  ⚠ Error {x}/{y}: {e}")
            return None

    img  = Image.open(cache)
    data = decode_terrarium(img)
    return x, y, data


async def download_all_tiles() -> list[tuple[int, int, int, np.ndarray]]:
    """Завантажити всі тайли для bbox."""
    west, south, east, north = BBOX

    x0, y1 = latlon_to_tile(north, west,  ZOOM)
    x1, y0 = latlon_to_tile(south, east,  ZOOM)

    tiles_xy = [
        (x, y)
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
    ]

    print(f"📥 Завантажуємо {len(tiles_xy)} тайлів (zoom={ZOOM})...")
    print(f"   BBox: {BBOX}")

    out_dir = OUTDIR / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    async with httpx.AsyncClient() as client:
        tasks = [download_tile(client, x, y, ZOOM, out_dir) for x, y in tiles_xy]
        done  = await asyncio.gather(*tasks)

    for item in done:
        if item is not None:
            x, y, data = item
            results.append((x, y, ZOOM, data))

    print(f"✅ Завантажено: {len(results)}/{len(tiles_xy)} тайлів")
    return results


def stitch_tiles(
    tiles: list[tuple[int, int, int, np.ndarray]],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Зшити тайли в один суцільний масив."""
    if not tiles:
        raise ValueError("Немає тайлів для зшивки")

    xs = sorted(set(t[0] for t in tiles))
    ys = sorted(set(t[1] for t in tiles))

    tile_h, tile_w = tiles[0][3].shape
    total_w = len(xs) * tile_w
    total_h = len(ys) * tile_h

    merged = np.full((total_h, total_w), np.nan, dtype=np.float32)

    tile_dict = {(t[0], t[1]): t[3] for t in tiles}
    z         = tiles[0][2]

    for row_i, y in enumerate(ys):
        for col_i, x in enumerate(xs):
            data = tile_dict.get((x, y))
            if data is not None:
                r0 = row_i * tile_h
                c0 = col_i * tile_w
                merged[r0:r0+tile_h, c0:c0+tile_w] = data

    # Загальний bbox
    x_min, x_max = xs[0],  xs[-1]
    y_min, y_max = ys[0],  ys[-1]
    west,  south, _,    _     = tile_to_bbox(x_min, y_max+1, z)
    _,     _,     east, north = tile_to_bbox(x_max, y_min,   z)

    return merged, (west, south, east, north)


def save_geotiff(
    data: np.ndarray,
    bbox: tuple[float, float, float, float],
    path: Path,
) -> None:
    """Зберегти як GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_bounds

        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = data.shape
        west, south, east, north = bbox
        transform = from_bounds(west, south, east, north, w, h)
        nodata_val = np.where(np.isnan(data), -9999.0, data)

        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=h, width=w, count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
            compress="lzw",
        ) as dst:
            dst.write(nodata_val.astype(np.float32), 1)

        print(f"✅ GeoTIFF збережено: {path}")
        print(f"   Розмір: {w}×{h} пікс")
        print(f"   BBox: {west:.3f},{south:.3f},{east:.3f},{north:.3f}")

        valid = data[~np.isnan(data)]
        print(f"   Висоти: {valid.min():.0f}..{valid.max():.0f} м")
        print(f"   Coverage: {len(valid)/data.size*100:.1f}%")

    except ImportError:
        print("⚠ rasterio не встановлено — зберігаємо як numpy .npy")
        np.save(path.with_suffix(".npy"), data)


def save_preview(
    data: np.ndarray,
    path: Path,
) -> None:
    """Зберегти PNG preview hillshade."""
    from scipy.ndimage import sobel

    clean = np.where(np.isnan(data), float(np.nanmin(data)), data)

    # Простий hillshade
    dx = sobel(clean.astype(np.float64), axis=1)
    dy = sobel(clean.astype(np.float64), axis=0)

    import math
    az   = math.radians(315)
    alt  = math.radians(45)
    hs   = (math.cos(alt) * (1 / (1 + dx**2 + dy**2)**0.5)
            + math.sin(alt) * (-dx * math.cos(az) - dy * math.sin(az))
              / (1 + dx**2 + dy**2)**0.5)
    hs   = np.clip((hs + 1) / 2 * 255, 0, 255).astype(np.uint8)

    # Elevation colormap
    norm = (clean - clean.min()) / max(clean.max() - clean.min(), 1)
    r    = np.clip(norm * 200 + 55,  0, 255).astype(np.uint8)
    g    = np.clip((1-abs(norm-0.5)*2) * 180 + 40, 0, 255).astype(np.uint8)
    b    = np.clip((1-norm) * 150 + 30, 0, 255).astype(np.uint8)

    # Blend з hillshade
    factor = (hs / 255.0)[:, :, np.newaxis]
    rgb    = np.stack([r, g, b], axis=-1)
    blended = np.clip(rgb * factor * 1.5, 0, 255).astype(np.uint8)

    img = Image.fromarray(blended, "RGB")
    img.save(path)
    print(f"✅ Preview збережено: {path} ({img.width}×{img.height})")


async def main() -> None:
    print("=" * 55)
    print("◈  GeoEngine — Seed Data Downloader")
    print("=" * 55)

    # 1. Завантажити тайли
    tiles = await download_all_tiles()
    if not tiles:
        print("❌ Не вдалося завантажити жодного тайлу")
        return

    # 2. Зшити в один масив
    print("\n🔗 Зшиваємо тайли...")
    merged, bbox = stitch_tiles(tiles)
    print(f"   Merged: {merged.shape[1]}×{merged.shape[0]} пікс")

    # 3. Зберегти GeoTIFF
    print("\n💾 Зберігаємо...")
    save_geotiff(merged, bbox, OUTDIR / "dem" / "carpathians.tif")

    # 4. Preview
    save_preview(merged, OUTDIR / "dem" / "carpathians_preview.png")

    print("\n" + "=" * 55)
    print("✅ Seed дані готові!")
    print(f"   GeoTIFF:  {OUTDIR}/dem/carpathians.tif")
    print(f"   Preview:  {OUTDIR}/dem/carpathians_preview.png")
    print(f"   Tiles:    {OUTDIR}/tiles/terrarium/")
    print("=" * 55)
    print("\nНаступний крок:")
    print("  python scripts/test_mesh.py")


if __name__ == "__main__":
    asyncio.run(main())
