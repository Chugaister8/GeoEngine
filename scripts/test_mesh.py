"""
GeoEngine — Test Mesh Pipeline
Перевіряє що весь Python pipeline працює від DEM до mesh.

Запуск:
    python scripts/test_mesh.py

Очікуваний результат:
    ✅ DEM завантажено
    ✅ Mesh побудовано
    ✅ GLB збережено
    ✅ Stats виведено
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Додаємо корінь проекту в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core-python"))


def check_dem() -> object:
    """Крок 1: Завантажити DEM."""
    from geoengine.dem.loader import DEMLoader
    from geoengine.geo.bbox   import BBox

    dem_path = Path("data/dem/carpathians.tif")
    if not dem_path.exists():
        print("❌ DEM файл не знайдено.")
        print("   Спочатку запусти: python scripts/seed_data.py")
        sys.exit(1)

    print("📂 Завантаження DEM...")
    t0     = time.perf_counter()
    loader = DEMLoader()
    tile   = loader.load(dem_path)
    elapsed = time.perf_counter() - t0

    print(f"   ✅ {tile.width}×{tile.height} пікс за {elapsed*1000:.0f}мс")
    print(f"   Висоти: {tile.min_elevation:.0f}..{tile.max_elevation:.0f} м")
    print(f"   Coverage: {tile.coverage_pct:.1f}%")
    print(f"   BBox: {tile.bbox}")
    return tile


def check_analysis(tile: object) -> None:
    """Крок 2: Аналіз рельєфу."""
    from geoengine.dem.analysis import (
        compute_slope, compute_hillshade, compute_contours
    )

    print("\n📐 Аналіз рельєфу...")

    t0    = time.perf_counter()
    slope = compute_slope(tile)
    print(f"   ✅ Slope: середнє={slope.mean_slope_deg:.1f}°  "
          f"макс={slope.max_slope_deg:.1f}°  "
          f"({(time.perf_counter()-t0)*1000:.0f}мс)")

    t0 = time.perf_counter()
    hs = compute_hillshade(tile)
    print(f"   ✅ Hillshade: {hs.data.min():.0f}..{hs.data.max():.0f}  "
          f"({(time.perf_counter()-t0)*1000:.0f}мс)")

    t0       = time.perf_counter()
    contours = compute_contours(tile, interval=200.0)
    print(f"   ✅ Contours: {len(contours.lines)} ізоліній (крок 200м)  "
          f"({(time.perf_counter()-t0)*1000:.0f}мс)")


def check_mesh(tile: object) -> object:
    """Крок 3: Побудова mesh."""
    from geoengine.mesh.terrain import TerrainMeshBuilder
    from geoengine.geo.coords   import LLH

    print("\n🏗  Побудова 3D mesh...")
    c       = tile.bbox.center
    origin  = LLH(lat=c[0], lon=c[1])

    for max_v, label in [(4096, "LOD-fast"), (32768, "LOD-normal")]:
        t0      = time.perf_counter()
        builder = TerrainMeshBuilder(origin=origin, skirt_height=200.0)
        mesh    = builder.build(tile, max_verts=max_v, lod_level=0)
        elapsed = time.perf_counter() - t0

        print(f"   ✅ {label}: {mesh.vertex_count:>7,} verts  "
              f"{mesh.triangle_count:>7,} tris  "
              f"{mesh.memory_mb:>5.1f}MB  "
              f"{elapsed*1000:>5.0f}мс")

    return mesh


def check_normals(tile: object) -> None:
    """Крок 4: Normal map."""
    from geoengine.mesh.normals import generate_normal_map
    import numpy as np

    print("\n🗺  Normal map...")
    t0 = time.perf_counter()
    nm = generate_normal_map(tile)
    elapsed = time.perf_counter() - t0

    print(f"   ✅ Shape: {nm.shape}  dtype: {nm.dtype}  "
          f"range: {nm.min()}..{nm.max()}  "
          f"({elapsed*1000:.0f}мс)")

    # Перевірка: G-канал (Up) > 100 у середньому
    avg_g = float(nm[:,:,1].mean())
    status = "✅" if avg_g > 100 else "⚠️"
    print(f"   {status} G-канал середнє: {avg_g:.1f} (норма > 100)")


def check_export(mesh: object, tile: object) -> None:
    """Крок 5: Експорт."""
    from geoengine.io.gltf    import terrain_to_gltf
    from geoengine.io.geotiff import dem_tile_to_geotiff
    from geoengine.io.geojson import contours_to_geojson
    from geoengine.dem.analysis import compute_contours

    out = Path("data/output")
    out.mkdir(parents=True, exist_ok=True)

    print("\n💾 Експорт...")

    # GLB
    t0  = time.perf_counter()
    glb = terrain_to_gltf(mesh, out / "carpathians.glb", format="glb")
    sz  = glb.stat().st_size / 1024 / 1024
    print(f"   ✅ GLB: {glb.name}  {sz:.1f}MB  ({(time.perf_counter()-t0)*1000:.0f}мс)")

    # GeoTIFF
    t0  = time.perf_counter()
    tif = out / "carpathians_out.tif"
    dem_tile_to_geotiff(tile, tif)
    sz  = tif.stat().st_size / 1024 / 1024
    print(f"   ✅ GeoTIFF: {tif.name}  {sz:.1f}MB  ({(time.perf_counter()-t0)*1000:.0f}мс)")

    # Contours GeoJSON
    t0       = time.perf_counter()
    contours = compute_contours(tile, interval=200.0)
    geo_path = out / "contours.geojson"
    contours_to_geojson(contours, geo_path)
    sz = geo_path.stat().st_size / 1024
    print(f"   ✅ GeoJSON: {geo_path.name}  {sz:.0f}KB  "
          f"({len(contours.lines)} ліній)  ({(time.perf_counter()-t0)*1000:.0f}мс)")


def check_scene(tile: object) -> None:
    """Крок 6: Scene graph."""
    from geoengine.scene import Scene

    print("\n🎬 Scene graph...")
    scene = Scene(name="Карпати — тест")
    node  = scene.add_terrain(tile)
    scene.fly_to(lat=48.5, lon=24.2, alt=8000)
    scene.set_time(hours=14.0)

    d = scene.to_dict()
    print(f"   ✅ Scene: nodes={scene.node_count}  layers={scene.layers.layer_count}")
    print(f"   ✅ Camera: {d['camera']['lat']:.2f}°N {d['camera']['lon']:.2f}°E  alt={d['camera']['alt']:.0f}м")
    print(f"   ✅ Sun direction: {[round(x,3) for x in d['sun']['direction']]}")
    print(f"   ✅ to_dict() keys: {list(d.keys())}")


def check_websocket_payload(mesh: object) -> None:
    """Крок 7: WS серіалізація."""
    import base64, json

    print("\n📡 WebSocket payload...")
    t0   = time.perf_counter()
    data = mesh.to_dict()
    elapsed = time.perf_counter() - t0

    # Перевірка структури
    required = ["type", "vertex_count", "triangle_count", "bbox", "buffers"]
    missing  = [k for k in required if k not in data]
    if missing:
        print(f"   ❌ Відсутні ключі: {missing}")
    else:
        print(f"   ✅ Структура OK: {required}")

    # Перевірка буферів
    for buf_name in ["vertices", "indices", "uvs", "normals"]:
        b64 = data["buffers"][buf_name]
        raw = base64.b64decode(b64)
        print(f"   ✅ {buf_name}: {len(raw):,} bytes ({len(raw)/1024:.1f}KB)")

    # JSON розмір
    json_str = json.dumps(data)
    print(f"   ✅ JSON розмір: {len(json_str)/1024:.1f}KB  "
          f"(серіалізація: {elapsed*1000:.0f}мс)")


def main() -> None:
    print("=" * 60)
    print("◈  GeoEngine — Pipeline Test")
    print("=" * 60)

    total_start = time.perf_counter()
    errors: list[str] = []

    try:
        tile = check_dem()
    except Exception as e:
        print(f"❌ КРОК 1 FAILED: {e}")
        sys.exit(1)

    try:
        check_analysis(tile)
    except Exception as e:
        errors.append(f"Analysis: {e}")
        print(f"   ⚠️ {e}")

    try:
        mesh = check_mesh(tile)
    except Exception as e:
        print(f"❌ КРОК 3 FAILED: {e}")
        sys.exit(1)

    try:
        check_normals(tile)
    except Exception as e:
        errors.append(f"Normals: {e}")
        print(f"   ⚠️ {e}")

    try:
        check_export(mesh, tile)
    except Exception as e:
        errors.append(f"Export: {e}")
        print(f"   ⚠️ {e}")

    try:
        check_scene(tile)
    except Exception as e:
        errors.append(f"Scene: {e}")
        print(f"   ⚠️ {e}")

    try:
        check_websocket_payload(mesh)
    except Exception as e:
        errors.append(f"WS payload: {e}")
        print(f"   ⚠️ {e}")

    total = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    if errors:
        print(f"⚠️  Завершено з {len(errors)} попередженнями за {total:.1f}с")
        for e in errors:
            print(f"   • {e}")
    else:
        print(f"✅ ВСІ КРОКИ ПРОЙШЛИ за {total:.1f}с")
    print("=" * 60)

    if not errors:
        print("\n🎉 Python pipeline повністю робочий!")
        print("   Наступний крок: python scripts/test_server.py")


if __name__ == "__main__":
    main()
