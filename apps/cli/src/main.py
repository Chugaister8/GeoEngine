"""
GeoEngine — CLI
Командний рядок для роботи з GeoEngine без Python коду.

Команди:
  terrain   — завантаження та обробка DEM
  osm       — завантаження OSM даних
  analysis  — GIS аналіз
  export    — експорт у різні формати
  scene     — управління сценою
  server    — запуск сервера

Usage:
  geoengine terrain fetch --bbox 22,47,25,50 --source copernicus25
  geoengine osm buildings --bbox 23,48,23.1,48.1
  geoengine analysis slope --input dem.tif --output slope.tif
  geoengine export gltf --input dem.tif --output model.glb
  geoengine server --port 8000
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table   import Table
from rich.panel   import Panel
from rich import print as rprint

# ----------------------------------------------------------------
# TYPER APP
# ----------------------------------------------------------------

app = Console()

cli = typer.Typer(
    name="geoengine",
    help="🌍 GeoEngine — 3D Геопросторовий Рушій",
    add_completion=True,
    rich_markup_mode="rich",
)

terrain_app  = typer.Typer(help="Terrain (DEM) операції")
osm_app      = typer.Typer(help="OpenStreetMap дані")
analysis_app = typer.Typer(help="GIS аналітика")
export_app   = typer.Typer(help="Експорт даних")
server_app   = typer.Typer(help="Сервер управління")

cli.add_typer(terrain_app,  name="terrain")
cli.add_typer(osm_app,      name="osm")
cli.add_typer(analysis_app, name="analysis")
cli.add_typer(export_app,   name="export")
cli.add_typer(server_app,   name="server")

console = Console()


# ----------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------

def parse_bbox(bbox_str: str):
    """Парсинг bbox рядку "west,south,east,north"."""
    from geoengine.geo.bbox import BBox
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return BBox(west=parts[0], south=parts[1], east=parts[2], north=parts[3])
    except ValueError:
        console.print(
            f"[red]❌ Невірний формат bbox: {bbox_str!r}[/red]\n"
            "Очікується: west,south,east,north (наприклад: 22.0,47.5,25.0,49.5)"
        )
        raise typer.Exit(code=1)


def success(msg: str) -> None:
    console.print(f"[green]✅ {msg}[/green]")


def error(msg: str) -> None:
    console.print(f"[red]❌ {msg}[/red]")
    raise typer.Exit(code=1)


def info(msg: str) -> None:
    console.print(f"[cyan]ℹ  {msg}[/cyan]")


# ----------------------------------------------------------------
# TERRAIN COMMANDS
# ----------------------------------------------------------------

@terrain_app.command("fetch")
def terrain_fetch(
    bbox:    str = typer.Option(..., "--bbox", "-b",
        help="BBox: west,south,east,north (градуси)"),
    source:  str = typer.Option("copernicus25", "--source", "-s",
        help="DEM джерело: srtm30 | copernicus25 | terrarium"),
    output:  Optional[Path] = typer.Option(None, "--output", "-o",
        help="Вихідний GeoTIFF файл (за замовчуванням: dem_{bbox}.tif)"),
    zoom:    int = typer.Option(10, "--zoom", "-z",
        help="Zoom рівень для тайлових джерел"),
):
    """
    📥 Завантажити DEM дані для bbox.

    Приклад:
      geoengine terrain fetch --bbox 22,47,25,50 --source copernicus25
    """
    from geoengine.dem.sources import DEMSourceManager, DEMSourceID

    b = parse_bbox(bbox)

    if output is None:
        name   = f"dem_{b.west:.2f}_{b.south:.2f}_{b.east:.2f}_{b.north:.2f}.tif"
        output = Path(name)

    info(f"Завантаження DEM: {source} для {b}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Завантаження...", total=None)

        async def fetch():
            manager  = DEMSourceManager()
            dem_tile = await manager.fetch(bbox=b, source=DEMSourceID(source))
            return dem_tile

        dem_tile = asyncio.run(fetch())
        progress.update(task, description="Зберігання...")

        from geoengine.io.geotiff import dem_tile_to_geotiff
        dem_tile_to_geotiff(dem_tile, output)

    success(f"Збережено: {output}")
    console.print(f"  Розмір:   {dem_tile.width}×{dem_tile.height} пікс")
    console.print(f"  Висоти:   {dem_tile.min_elevation:.0f}..{dem_tile.max_elevation:.0f} м")
    console.print(f"  Coverage: {dem_tile.coverage_pct:.1f}%")


@terrain_app.command("info")
def terrain_info(
    input: Path = typer.Argument(..., help="Шлях до GeoTIFF файлу"),
):
    """
    📊 Показати метадані GeoTIFF файлу.
    """
    from geoengine.io.geotiff import read_geotiff_meta
    from geoengine.dem.loader import DEMLoader

    if not input.exists():
        error(f"Файл не знайдено: {input}")

    meta   = read_geotiff_meta(input)
    loader = DEMLoader()
    tile   = loader.load(input)

    t = Table(title=f"📄 {input.name}", show_header=False)
    t.add_column("Поле",      style="cyan")
    t.add_column("Значення",  style="white")

    t.add_row("Розмір",    f"{meta['width']}×{meta['height']} пікс")
    t.add_row("CRS",       meta["crs"])
    t.add_row("Тип даних", meta["dtype"])
    t.add_row("NoData",    str(meta["nodata"]))
    t.add_row("BBox",      f"W={meta['bounds']['west']:.4f} S={meta['bounds']['south']:.4f} "
                           f"E={meta['bounds']['east']:.4f} N={meta['bounds']['north']:.4f}")
    t.add_row("Мін. висота", f"{tile.min_elevation:.1f} м")
    t.add_row("Макс. висота",f"{tile.max_elevation:.1f} м")
    t.add_row("Сер. висота", f"{tile.mean_elevation:.1f} м")
    t.add_row("Coverage",   f"{tile.coverage_pct:.1f}%")
    t.add_row("Resolution X", f"{tile.resolution_x:.6f}°/піксель")

    console.print(t)


@terrain_app.command("merge")
def terrain_merge(
    inputs: list[Path] = typer.Argument(..., help="GeoTIFF файли для злиття"),
    output: Path = typer.Option(..., "--output", "-o", help="Вихідний файл"),
    method: str  = typer.Option("first", "--method", "-m",
        help="Метод: first | last | mean | max | min"),
):
    """
    🔗 Злити кілька GeoTIFF файлів в один.
    """
    from geoengine.dem.loader   import DEMLoader
    from geoengine.dem.processor import merge_tiles
    from geoengine.io.geotiff   import dem_tile_to_geotiff

    loader = DEMLoader()
    tiles  = []

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), console=console,
    ) as progress:
        task = progress.add_task("Завантаження...", total=len(inputs))
        for path in inputs:
            tiles.append(loader.load(path))
            progress.advance(task)

        progress.update(task, description="Злиття...")
        merged = merge_tiles(tiles, method=method)

        progress.update(task, description="Зберігання...")
        dem_tile_to_geotiff(merged, output)

    success(f"Злито {len(inputs)} файлів → {output}")
    console.print(f"  Розмір: {merged.width}×{merged.height}")


# ----------------------------------------------------------------
# OSM COMMANDS
# ----------------------------------------------------------------

@osm_app.command("buildings")
def osm_buildings(
    bbox:   str  = typer.Option(..., "--bbox", "-b", help="BBox: west,south,east,north"),
    output: Path = typer.Option(Path("buildings.geojson"), "--output", "-o"),
    format: str  = typer.Option("geojson", "--format", "-f",
        help="Формат: geojson | glb"),
):
    """
    🏢 Завантажити будівлі OSM та зберегти.
    """
    from geoengine.osm.fetcher import OverpassFetcher, OverpassQuery

    b = parse_bbox(bbox)
    info(f"Завантаження будівель для {b}")

    async def fetch():
        fetcher = OverpassFetcher()
        return await fetcher.fetch_buildings(b)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Overpass API...", total=None)
        osm  = asyncio.run(fetch())
        progress.update(task, description="Генерація 3D...")

    from geoengine.geo.coords import LLH
    c      = b.center
    origin = LLH(lat=c[0], lon=c[1])

    if format == "geojson":
        from geoengine.io.geojson import write_geojson
        features = []
        for way in osm.buildings():
            if way.coords:
                coords = [[lon, lat] for lat, lon in way.coords]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": dict(way.tags),
                })
        write_geojson(output, features, name="buildings")
        success(f"Збережено {len(features)} будівель → {output}")

    elif format == "glb":
        from geoengine.osm.buildings import BuildingExtruder
        from geoengine.io.gltf import buildings_to_gltf

        extruder   = BuildingExtruder(origin_lat=origin.lat, origin_lon=origin.lon)
        collection = extruder.extrude_all(osm)
        glb_path   = output.with_suffix(".glb")
        buildings_to_gltf(collection, glb_path)
        success(f"Збережено {len(collection.meshes)} будівель → {glb_path}")


# ----------------------------------------------------------------
# ANALYSIS COMMANDS
# ----------------------------------------------------------------

@analysis_app.command("slope")
def analysis_slope(
    input:  Path = typer.Argument(..., help="Вхідний GeoTIFF"),
    output: Path = typer.Option(Path("slope.tif"), "--output", "-o"),
    unit:   str  = typer.Option("degrees", "--unit",
        help="Одиниці: degrees | percent"),
):
    """
    📐 Аналіз крутизни схилів.
    """
    from geoengine.dem.loader    import DEMLoader
    from geoengine.dem.analysis  import compute_slope
    from geoengine.io.geotiff    import write_geotiff

    loader = DEMLoader()
    tile   = loader.load(input)
    result = compute_slope(tile)

    data   = result.degrees if unit == "degrees" else result.percent
    write_geotiff(output, data, tile.bbox, nodata=-9999.0)

    success(f"Slope збережено → {output}")
    console.print(f"  Середнє: {result.mean_slope_deg:.1f}°")
    console.print(f"  Максимум: {result.max_slope_deg:.1f}°")


@analysis_app.command("contours")
def analysis_contours(
    input:    Path  = typer.Argument(..., help="Вхідний GeoTIFF"),
    output:   Path  = typer.Option(Path("contours.geojson"), "--output", "-o"),
    interval: float = typer.Option(100.0, "--interval", "-i",
        help="Крок ізоліній (метри)"),
):
    """
    〰 Генерація ізоліній рельєфу.
    """
    from geoengine.dem.loader   import DEMLoader
    from geoengine.dem.analysis import compute_contours
    from geoengine.io.geojson   import contours_to_geojson

    loader = DEMLoader()
    tile   = loader.load(input)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Генерація ізоліній...", total=None)
        result = compute_contours(tile, interval=interval)

    contours_to_geojson(result, output)
    success(f"Збережено {len(result.lines)} ізоліній → {output}")


@analysis_app.command("hillshade")
def analysis_hillshade(
    input:    Path  = typer.Argument(..., help="Вхідний GeoTIFF"),
    output:   Path  = typer.Option(Path("hillshade.tif"), "--output", "-o"),
    azimuth:  float = typer.Option(315.0, "--azimuth", "-a",
        help="Азимут сонця (0=Пн, 90=Сх, 315=ПнЗх)"),
    altitude: float = typer.Option(45.0, "--altitude",
        help="Висота сонця над горизонтом (градуси)"),
):
    """
    ☀ Генерація hillshade (тіньове відмивання).
    """
    from geoengine.dem.loader   import DEMLoader
    from geoengine.dem.analysis import compute_hillshade
    from geoengine.io.geotiff   import write_geotiff

    loader = DEMLoader()
    tile   = loader.load(input)
    result = compute_hillshade(tile, azimuth=azimuth, altitude=altitude)
    write_geotiff(output, result.data, tile.bbox)

    success(f"Hillshade збережено → {output}")


# ----------------------------------------------------------------
# EXPORT COMMANDS
# ----------------------------------------------------------------

@export_app.command("gltf")
def export_gltf(
    input:  Path = typer.Argument(..., help="Вхідний GeoTIFF"),
    output: Path = typer.Option(Path("terrain.glb"), "--output", "-o"),
    verts:  int  = typer.Option(65536, "--max-verts",
        help="Максимальна кількість вершин"),
):
    """
    🎲 Експорт терейну у glTF/GLB формат.
    """
    from geoengine.dem.loader    import DEMLoader
    from geoengine.mesh.terrain  import TerrainMeshBuilder
    from geoengine.geo.coords    import LLH
    from geoengine.io.gltf       import terrain_to_gltf

    loader = DEMLoader()
    tile   = loader.load(input)
    c      = tile.bbox.center
    origin = LLH(lat=c[0], lon=c[1])

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Генерація меша...", total=None)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=200.0)
        mesh    = builder.build(tile, max_verts=verts)

    fmt = "glb" if output.suffix.lower() == ".glb" else "gltf"
    terrain_to_gltf(mesh, output, format=fmt)

    success(f"Збережено → {output}")
    console.print(f"  Вершин:    {mesh.vertex_count:,}")
    console.print(f"  Трикутнів: {mesh.triangle_count:,}")
    console.print(f"  Пам'ять:   {mesh.memory_mb:.1f} MB")


# ----------------------------------------------------------------
# SERVER COMMANDS
# ----------------------------------------------------------------

@server_app.command("start")
def server_start(
    host:  str = typer.Option("0.0.0.0", "--host", help="Хост"),
    port:  int = typer.Option(8000,       "--port", "-p", help="Порт"),
    debug: bool = typer.Option(False,     "--debug", help="Debug режим"),
    reload: bool = typer.Option(False,    "--reload", help="Auto-reload"),
):
    """
    🚀 Запустити GeoEngine сервер.
    """
    import uvicorn

    console.print(Panel(
        f"[cyan]GeoEngine Server[/cyan]\n"
        f"URL:   http://{host}:{port}\n"
        f"WS:    ws://{host}:{port}/ws\n"
        f"Docs:  http://{host}:{port}/docs\n"
        f"Debug: {debug}",
        title="🌍 GeoEngine",
    ))

    uvicorn.run(
        "apps.server.src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="debug" if debug else "info",
    )


# ----------------------------------------------------------------
# ROOT COMMANDS
# ----------------------------------------------------------------

@cli.command("version")
def version():
    """Показати версію GeoEngine."""
    from geoengine import __version__
    console.print(f"[cyan]GeoEngine[/cyan] v{__version__}")


@cli.command("info")
def cli_info():
    """Показати інформацію про встановлені компоненти."""
    from rich.table import Table

    t = Table(title="🌍 GeoEngine Info", show_header=True)
    t.add_column("Компонент",  style="cyan")
    t.add_column("Версія",     style="green")
    t.add_column("Статус",     style="white")

    components = [
        ("geoengine-core", "0.1.0",  "✅"),
        ("rasterio",        None,     None),
        ("numpy",           None,     None),
        ("fastapi",         None,     None),
        ("laspy",           None,     None),
    ]

    for name, version_str, status in components:
        try:
            import importlib
            mod = importlib.import_module(name.replace("-", "_"))
            v   = getattr(mod, "__version__", version_str or "?")
            s   = status or "✅"
        except ImportError:
            v = "—"
            s = "❌ не встановлено"
        t.add_row(name, v, s)

    console.print(t)


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
