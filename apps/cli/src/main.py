"""
GeoEngine CLI
Командний інтерфейс для роботи з DEM, mesh, аналітикою та симуляціями.

Встановлення:
    uv pip install -e apps/cli

Використання:
    geoengine --help
    geoengine dem download --bbox 22,47,24,49 --source terrarium
    geoengine dem info data/dem.tif
    geoengine mesh build data/dem.tif --output data/mesh.glb
    geoengine analysis slope data/dem.tif --output data/slope.tif
    geoengine sim ballistics --lat 48.5 --lon 24.2 --az 45 --el 30 --v0 800
    geoengine sim fire --lat 48.5 --lon 24.2 --duration 6
    geoengine server start --port 8000
    geoengine seed --area ukraine --zoom 9
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# ── App ───────────────────────────────────────────────────────────

app = typer.Typer(
    name="geoengine",
    help="🌍 GeoEngine — 3D Геопросторовий Рушій",
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=True,
)

dem_app      = typer.Typer(help="DEM операції (завантаження, інфо, злиття)")
mesh_app     = typer.Typer(help="3D Mesh генерація")
analysis_app = typer.Typer(help="GIS аналітика (схил, горизонталі, hillshade)")
sim_app      = typer.Typer(help="Симуляції (балістика, вогонь)")
server_app   = typer.Typer(help="FastAPI сервер")

app.add_typer(dem_app,      name="dem")
app.add_typer(mesh_app,     name="mesh")
app.add_typer(analysis_app, name="analysis")
app.add_typer(sim_app,      name="sim")
app.add_typer(server_app,   name="server")

console = Console()
err     = Console(stderr=True)


# ── Версія ────────────────────────────────────────────────────────

def _version_callback(value: bool) -> None:
    if value:
        console.print("[bold cyan]GeoEngine[/] [green]0.1.0[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version", "-v",
        callback=_version_callback,
        is_eager=True,
        help="Показати версію та вийти.",
    ),
) -> None:
    """🌍 GeoEngine — 3D Геопросторовий Рушій."""


# ════════════════════════════════════════════════════════════════
# DEM КОМАНДИ
# ════════════════════════════════════════════════════════════════

@dem_app.command("download")
def dem_download(
    bbox: str = typer.Option(
        ...,
        "--bbox",
        help="BBox у форматі west,south,east,north (градуси WGS84). Приклад: 22,47,24,49",
    ),
    source: str = typer.Option(
        "terrarium",
        "--source", "-s",
        help="DEM джерело: terrarium | srtm30 | srtm90 | copernicus25",
    ),
    zoom: int = typer.Option(
        10,
        "--zoom", "-z",
        help="Zoom рівень для Terrarium тайлів (7-14)",
        min=4,
        max=14,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Вихідний файл (GeoTIFF). За замовчуванням: dem_<bbox>.tif",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache",
        help="Каталог кешу. За замовчуванням: ~/.geoengine/dem_cache",
    ),
) -> None:
    """
    Завантажити DEM дані для географічного BBox.

    \b
    Приклад:
        geoengine dem download --bbox 22,47,24,49 --source terrarium --zoom 9
        geoengine dem download --bbox 30,50,32,52 --source srtm30
    """
    # Парсимо bbox
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        west, south, east, north = parts
    except ValueError:
        err.print("[red]❌ Невірний формат bbox. Використовуй: west,south,east,north[/]")
        raise typer.Exit(1)

    if output is None:
        output = Path(f"dem_{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}.tif")

    console.print(Panel(
        f"[cyan]Джерело:[/]   {source}\n"
        f"[cyan]BBox:[/]      {west:.4f}°, {south:.4f}° → {east:.4f}°, {north:.4f}°\n"
        f"[cyan]Zoom:[/]      {zoom}\n"
        f"[cyan]Вихід:[/]     {output}",
        title="[bold]📥 Завантаження DEM[/]",
        border_style="blue",
    ))

    async def _download() -> None:
        from geoengine.geo.bbox import BBox
        from geoengine.dem.sources import DEMSourceManager, DEMSourceID
        from geoengine.io.geotiff import write_geotiff

        bbox_obj = BBox(west=west, south=south, east=east, north=north)

        try:
            source_id = DEMSourceID(source)
        except ValueError:
            err.print(f"[red]❌ Невідоме джерело: {source!r}[/]")
            raise typer.Exit(1)

        mgr = DEMSourceManager(
            cache_dir=cache_dir or Path.home() / ".geoengine" / "dem_cache",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Завантаження тайлів...", total=None)

            tiles = await mgr.fetch_tiles(
                bbox=bbox_obj,
                zoom=zoom,
                source=source_id,
            )

            progress.update(task, description="Злиття тайлів...")

            if not tiles:
                err.print("[red]❌ Не вдалося завантажити жодного тайлу[/]")
                raise typer.Exit(1)

            from geoengine.dem.processor import merge_tiles, fill_gaps
            merged = tiles[0] if len(tiles) == 1 else merge_tiles(tiles)
            if merged.has_nodata:
                merged = fill_gaps(merged)

            progress.update(task, description="Збереження GeoTIFF...")
            write_geotiff(merged, output)
            progress.update(task, completed=True, description="Готово!")

        console.print(
            f"\n[green]✅ Збережено:[/] {output}\n"
            f"   Розмір: {merged.width}×{merged.height}px\n"
            f"   Висоти: {merged.min_elevation:.0f}м — {merged.max_elevation:.0f}м\n"
            f"   Покриття: {merged.coverage_pct:.1f}%"
        )

    asyncio.run(_download())


@dem_app.command("info")
def dem_info(
    path: Path = typer.Argument(..., help="Шлях до DEM файлу (GeoTIFF, HGT, ASC)"),
    full: bool = typer.Option(False, "--full", "-f", help="Показати повну статистику"),
) -> None:
    """
    Показати інформацію про DEM файл.

    \b
    Приклад:
        geoengine dem info data/dem.tif
        geoengine dem info data/dem.tif --full
    """
    if not path.exists():
        err.print(f"[red]❌ Файл не знайдено: {path}[/]")
        raise typer.Exit(1)

    from geoengine.dem.loader import DEMLoader

    with Progress(SpinnerColumn(), TextColumn("Завантаження..."), console=console) as p:
        p.add_task("load")
        loader = DEMLoader()
        tile   = loader.load(path)

    table = Table(title=f"📊 DEM: {path.name}", border_style="blue")
    table.add_column("Параметр",  style="cyan",  no_wrap=True)
    table.add_column("Значення",  style="white")

    table.add_row("Файл",        str(path))
    table.add_row("Розмір",      f"{tile.width} × {tile.height} пікселів")
    table.add_row("CRS",         tile.crs)
    table.add_row("Роздільність X", f"{tile.resolution_x:.6f}° ({tile.resolution_x * 111_320:.1f} м/пікс)")
    table.add_row("Роздільність Y", f"{tile.resolution_y:.6f}° ({tile.resolution_y * 111_320:.1f} м/пікс)")
    table.add_row("Мін. висота", f"{tile.min_elevation:.1f} м")
    table.add_row("Макс. висота",f"{tile.max_elevation:.1f} м")
    table.add_row("Середня",     f"{tile.mean_elevation:.1f} м")
    table.add_row("Покриття",    f"{tile.coverage_pct:.1f}%")
    table.add_row("BBox",
        f"W={tile.bbox.west:.4f}° S={tile.bbox.south:.4f}° "
        f"E={tile.bbox.east:.4f}° N={tile.bbox.north:.4f}°"
    )
    table.add_row("Площа",       f"{tile.bbox.area_m2 / 1e6:.1f} км²")

    if full:
        import numpy as np
        data = tile.data
        valid = data[~np.isnan(data)]
        if len(valid):
            table.add_row("Медіана",   f"{float(np.median(valid)):.1f} м")
            table.add_row("Std Dev",   f"{float(np.std(valid)):.1f} м")
            table.add_row("P10",       f"{float(np.percentile(valid, 10)):.1f} м")
            table.add_row("P90",       f"{float(np.percentile(valid, 90)):.1f} м")

    console.print(table)


@dem_app.command("merge")
def dem_merge(
    paths: list[Path] = typer.Argument(..., help="Шляхи до DEM файлів для злиття"),
    output: Path = typer.Option(
        Path("merged.tif"),
        "--output", "-o",
        help="Вихідний файл",
    ),
    method: str = typer.Option(
        "first",
        "--method", "-m",
        help="Метод overlap: first | last | mean | max | min",
    ),
) -> None:
    """
    Злити кілька DEM файлів в один.

    \b
    Приклад:
        geoengine dem merge tile1.tif tile2.tif tile3.tif -o merged.tif
        geoengine dem merge *.tif -o full_ukraine.tif --method mean
    """
    for p in paths:
        if not p.exists():
            err.print(f"[red]❌ Файл не знайдено: {p}[/]")
            raise typer.Exit(1)

    from geoengine.dem.loader   import DEMLoader
    from geoengine.dem.processor import merge_tiles
    from geoengine.io.geotiff   import write_geotiff

    console.print(f"[bold]🔗 Злиття {len(paths)} файлів...[/]")

    loader = DEMLoader()
    tiles  = [loader.load(p) for p in paths]

    merged = merge_tiles(tiles, method=method)  # type: ignore[arg-type]

    write_geotiff(merged, output)
    console.print(
        f"[green]✅ Збережено:[/] {output}\n"
        f"   Розмір: {merged.width}×{merged.height}px\n"
        f"   Висоти: {merged.min_elevation:.0f}м — {merged.max_elevation:.0f}м"
    )


# ════════════════════════════════════════════════════════════════
# MESH КОМАНДИ
# ════════════════════════════════════════════════════════════════

@mesh_app.command("build")
def mesh_build(
    dem_path: Path = typer.Argument(..., help="Шлях до DEM файлу"),
    output: Path = typer.Option(
        Path("terrain.glb"),
        "--output", "-o",
        help="Вихідний файл (.glb або .json)",
    ),
    max_verts: int = typer.Option(
        65_536,
        "--max-verts",
        help="Максимальна кількість вершин",
        min=64,
        max=262_144,
    ),
    method: str = typer.Option(
        "uniform",
        "--method", "-m",
        help="Метод: uniform | adaptive",
    ),
    skirt: float = typer.Option(
        200.0,
        "--skirt",
        help="Висота 'спідниці' для усунення щілин між тайлами (метри)",
    ),
    lod: int = typer.Option(
        0,
        "--lod",
        help="LOD рівень (0 = найдетальніший)",
        min=0,
        max=5,
    ),
) -> None:
    """
    Побудувати 3D mesh з DEM файлу.

    \b
    Приклад:
        geoengine mesh build data/dem.tif -o terrain.glb
        geoengine mesh build data/dem.tif -o terrain.glb --method adaptive --max-verts 32768
    """
    if not dem_path.exists():
        err.print(f"[red]❌ Файл не знайдено: {dem_path}[/]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[cyan]DEM:[/]       {dem_path}\n"
        f"[cyan]Вихід:[/]     {output}\n"
        f"[cyan]Метод:[/]     {method}\n"
        f"[cyan]Max вершин:[/] {max_verts:,}\n"
        f"[cyan]Skirt:[/]     {skirt} м",
        title="[bold]🏔 Побудова Mesh[/]",
        border_style="blue",
    ))

    from geoengine.dem.loader    import DEMLoader
    from geoengine.mesh.terrain  import TerrainMeshBuilder
    from geoengine.geo.coords    import LLH

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Завантаження DEM...")
        loader   = DEMLoader()
        dem_tile = loader.load(dem_path)

        p.update(task, description="Побудова mesh...")
        c = dem_tile.bbox.center
        origin  = LLH(lat=c[0], lon=c[1])
        builder = TerrainMeshBuilder(origin=origin, skirt_height=skirt)
        mesh    = builder.build(
            dem_tile,
            method=method,     # type: ignore[arg-type]
            max_verts=max_verts,
            lod_level=lod,
        )

        p.update(task, description="Збереження...")
        ext = output.suffix.lower()
        if ext == ".glb":
            from geoengine.io.gltf import GLTFBuilder
            builder_gltf = GLTFBuilder()
            builder_gltf.add_terrain_mesh(mesh)
            builder_gltf.save(output)
        elif ext == ".json":
            data = mesh.to_dict()
            output.write_text(json.dumps(data, indent=2))
        else:
            err.print(f"[yellow]⚠️  Невідоме розширення {ext!r}, зберігаю як JSON[/]")
            data = mesh.to_dict()
            output.with_suffix(".json").write_text(json.dumps(data, indent=2))

        p.update(task, description="Готово!")

    console.print(
        f"\n[green]✅ Збережено:[/] {output}\n"
        f"   Вершин:    {mesh.vertex_count:,}\n"
        f"   Трикутників: {mesh.triangle_count:,}\n"
        f"   Пам'ять:   {mesh.memory_mb:.1f} МБ\n"
        f"   LOD:       {lod}"
    )


@mesh_app.command("info")
def mesh_info(
    path: Path = typer.Argument(..., help="Шлях до mesh файлу (.glb або .json)"),
) -> None:
    """Показати інформацію про mesh файл."""
    if not path.exists():
        err.print(f"[red]❌ Файл не знайдено: {path}[/]")
        raise typer.Exit(1)

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        table = Table(title=f"📊 Mesh: {path.name}", border_style="blue")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значення", style="white")
        for k, v in data.items():
            if k != "buffers":
                table.add_row(k, str(v))
        console.print(table)
    else:
        size_mb = path.stat().st_size / 1024 / 1024
        console.print(f"[cyan]{path.name}[/] — {size_mb:.2f} МБ")


# ════════════════════════════════════════════════════════════════
# ANALYSIS КОМАНДИ
# ════════════════════════════════════════════════════════════════

@analysis_app.command("slope")
def analysis_slope(
    dem_path: Path = typer.Argument(..., help="Шлях до DEM файлу"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    degrees: bool = typer.Option(True, "--degrees/--percent", help="Градуси чи відсотки"),
) -> None:
    """
    Обчислити карту крутизни схилів.

    \b
    Приклад:
        geoengine analysis slope data/dem.tif -o slope.tif
    """
    if not dem_path.exists():
        err.print(f"[red]❌ Файл не знайдено: {dem_path}[/]")
        raise typer.Exit(1)

    if output is None:
        output = dem_path.with_stem(dem_path.stem + "_slope")

    from geoengine.dem.loader   import DEMLoader
    from geoengine.dem.analysis import compute_slope
    from geoengine.io.geotiff   import write_geotiff, DEMTile
    from rasterio.transform     import from_bounds

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Завантаження DEM...")
        loader   = DEMLoader()
        dem_tile = loader.load(dem_path)

        p.update(task, description="Обчислення схилів...")
        result = compute_slope(dem_tile)

        p.update(task, description="Збереження...")
        data = result.degrees if degrees else result.percent
        out_tile = DEMTile(
            data=data,
            bbox=dem_tile.bbox,
            transform=dem_tile.transform,
            crs=dem_tile.crs,
            source=str(output),
        )
        write_geotiff(out_tile, output)
        p.update(task, description="Готово!")

    console.print(
        f"[green]✅ Збережено:[/] {output}\n"
        f"   Середній схил: {result.mean_slope_deg:.1f}°\n"
        f"   Максимальний:  {result.max_slope_deg:.1f}°"
    )


@analysis_app.command("hillshade")
def analysis_hillshade(
    dem_path: Path = typer.Argument(..., help="Шлях до DEM файлу"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    azimuth: float = typer.Option(315.0, "--azimuth", help="Азимут сонця (0-360°)"),
    altitude: float = typer.Option(45.0, "--altitude", help="Кут сонця (0-90°)"),
    z_factor: float = typer.Option(1.0, "--z-factor", help="Вертикальне перебільшення"),
) -> None:
    """
    Побудувати карту тіньового рельєфу (hillshade).

    \b
    Приклад:
        geoengine analysis hillshade data/dem.tif -o hillshade.tif
        geoengine analysis hillshade data/dem.tif --azimuth 270 --altitude 30
    """
    if not dem_path.exists():
        err.print(f"[red]❌ Файл не знайдено: {dem_path}[/]")
        raise typer.Exit(1)

    if output is None:
        output = dem_path.with_stem(dem_path.stem + "_hillshade")

    from geoengine.dem.loader   import DEMLoader, DEMTile
    from geoengine.dem.analysis import compute_hillshade
    from geoengine.io.geotiff   import write_geotiff

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Завантаження DEM...")
        loader   = DEMLoader()
        dem_tile = loader.load(dem_path)

        p.update(task, description="Обчислення hillshade...")
        result = compute_hillshade(dem_tile, azimuth=azimuth, altitude=altitude, z_factor=z_factor)

        p.update(task, description="Збереження...")
        out_tile = DEMTile(
            data=result.data,
            bbox=dem_tile.bbox,
            transform=dem_tile.transform,
            crs=dem_tile.crs,
            source=str(output),
        )
        write_geotiff(out_tile, output)
        p.update(task, description="Готово!")

    console.print(
        f"[green]✅ Hillshade збережено:[/] {output}\n"
        f"   Азимут: {azimuth}°, Кут: {altitude}°, Z: {z_factor}"
    )


@analysis_app.command("contours")
def analysis_contours(
    dem_path: Path = typer.Argument(..., help="Шлях до DEM файлу"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Вихідний GeoJSON"),
    interval: float = typer.Option(100.0, "--interval", "-i", help="Інтервал між горизонталями (метри)"),
    base: float = typer.Option(0.0, "--base", help="Базова висота (метри)"),
) -> None:
    """
    Побудувати горизонталі (ізолінії рельєфу).

    \b
    Приклад:
        geoengine analysis contours data/dem.tif -o contours.geojson --interval 50
    """
    if not dem_path.exists():
        err.print(f"[red]❌ Файл не знайдено: {dem_path}[/]")
        raise typer.Exit(1)

    if output is None:
        output = dem_path.with_stem(dem_path.stem + "_contours").with_suffix(".geojson")

    from geoengine.dem.loader   import DEMLoader
    from geoengine.dem.analysis import compute_contours
    from geoengine.io.geojson   import contours_to_geojson

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Завантаження DEM...")
        loader   = DEMLoader()
        dem_tile = loader.load(dem_path)

        p.update(task, description="Обчислення горизонталей...")
        result = compute_contours(dem_tile, interval=interval, base=base)

        p.update(task, description="Збереження GeoJSON...")
        geojson = contours_to_geojson(result)
        output.write_text(json.dumps(geojson, indent=2))
        p.update(task, description="Готово!")

    console.print(
        f"[green]✅ Горизонталі збережено:[/] {output}\n"
        f"   Ліній: {len(result.lines)}\n"
        f"   Інтервал: {interval} м\n"
        f"   Розмір файлу: {output.stat().st_size / 1024:.1f} КБ"
    )


# ════════════════════════════════════════════════════════════════
# SIMULATION КОМАНДИ
# ════════════════════════════════════════════════════════════════

@sim_app.command("ballistics")
def sim_ballistics(
    lat: float = typer.Option(..., "--lat", help="Широта точки пострілу"),
    lon: float = typer.Option(..., "--lon", help="Довгота точки пострілу"),
    alt: float = typer.Option(0.0, "--alt", help="Висота точки пострілу (м)"),
    azimuth: float = typer.Option(..., "--az", help="Азимут пострілу (0=Північ, 90=Схід)"),
    elevation: float = typer.Option(..., "--el", help="Кут підвищення (градуси)"),
    v0: float = typer.Option(..., "--v0", help="Початкова швидкість (м/с)"),
    preset: str = typer.Option(
        "artillery_122mm",
        "--preset", "-p",
        help="Пресет снаряду: artillery_122mm | artillery_152mm | mortar_120mm | rifle_762x54 | bomb_250kg",
    ),
    wind_speed: float = typer.Option(0.0, "--wind", help="Швидкість вітру (м/с)"),
    wind_dir: float = typer.Option(0.0, "--wind-dir", help="Напрямок вітру (°, звідки дме)"),
    dem: Optional[Path] = typer.Option(None, "--dem", help="DEM файл для terrain hit detection"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Зберегти траєкторію як GeoJSON"),
    table: bool = typer.Option(False, "--table", "-t", help="Показати таблицю стрільби"),
) -> None:
    """
    Балістичний розрахунок траєкторії снаряду.

    \b
    Приклад:
        geoengine sim ballistics --lat 48.5 --lon 24.2 --az 45 --el 30 --v0 800
        geoengine sim ballistics --lat 48.5 --lon 24.2 --az 0 --el 45 --v0 800 \\
            --preset artillery_152mm --wind 5 --wind-dir 270 --dem data/dem.tif
    """
    from geoengine.simulation.ballistics import (
        BallisticsSolver, LatLonAlt, WindVector, ProjectileParams, BALLISTIC_PRESETS
    )

    # Перевірка пресету
    if preset not in BALLISTIC_PRESETS:
        err.print(f"[red]❌ Невідомий пресет: {preset!r}[/]")
        err.print(f"Доступні: {', '.join(BALLISTIC_PRESETS.keys())}")
        raise typer.Exit(1)

    console.print(Panel(
        f"[cyan]Позиція:[/]    {lat:.4f}°N, {lon:.4f}°E, {alt:.0f}м\n"
        f"[cyan]Азимут:[/]     {azimuth}°\n"
        f"[cyan]Підвищення:[/] {elevation}°\n"
        f"[cyan]Швидкість:[/]  {v0} м/с\n"
        f"[cyan]Снаряд:[/]     {preset}\n"
        f"[cyan]Вітер:[/]      {wind_speed} м/с @ {wind_dir}°",
        title="[bold]💥 Балістичний Розрахунок[/]",
        border_style="red",
    ))

    # Завантаження DEM
    dem_tile = None
    if dem and dem.exists():
        from geoengine.dem.loader import DEMLoader
        dem_tile = DEMLoader().load(dem)
        console.print(f"[dim]  DEM: {dem.name} ({dem_tile.width}×{dem_tile.height})[/]")

    wind = WindVector(speed_ms=wind_speed, direction_deg=wind_dir)
    solver = BallisticsSolver(dem_tile=dem_tile)

    with Progress(SpinnerColumn(), TextColumn("Розрахунок..."), console=console) as p:
        p.add_task("solve")
        result = solver.solve(
            origin=LatLonAlt(lat=lat, lon=lon, alt=alt),
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            muzzle_velocity=v0,
            projectile=preset,
            wind=wind,
        )

    # Результат
    impact = result.impact_point
    result_table = Table(title="📊 Результат", border_style="red")
    result_table.add_column("Параметр",  style="cyan")
    result_table.add_column("Значення",  style="white")

    result_table.add_row("Дальність",      f"{result.max_range_m:,.0f} м")
    result_table.add_row("Час польоту",    f"{result.flight_time_s:.2f} с")
    result_table.add_row("Макс. висота",   f"{result.max_height_m:,.0f} м")
    result_table.add_row("Швидкість удару",f"{result.impact_velocity:.1f} м/с")
    result_table.add_row("Кут падіння",    f"{result.impact_angle_deg:.1f}°")

    if impact:
        result_table.add_row("Точка падіння",
            f"{impact.lat:.6f}°N, {impact.lon:.6f}°E, {impact.alt:.0f}м")
    result_table.add_row("Рельєф",  "✅ Потрапив" if result.hit_terrain else "❌ Не торкнувся")

    console.print(result_table)

    # Таблиця стрільби
    if table:
        console.print("\n[bold]📋 Таблиця стрільби:[/]")
        shot_table = Table(border_style="dim")
        shot_table.add_column("Кут (°)", style="cyan")
        shot_table.add_column("Дальність (м)", style="white")
        shot_table.add_column("Час (с)", style="white")
        shot_table.add_column("Висота (м)", style="white")

        entries = solver.solve_range_table(
            origin=LatLonAlt(lat=lat, lon=lon, alt=alt),
            muzzle_velocity=v0,
            projectile=preset,
            wind=wind,
        )
        for e in entries:
            shot_table.add_row(
                str(e["elevation_deg"]),
                f"{e['range_m']:,.0f}",
                f"{e['flight_time_s']:.1f}",
                f"{e['max_height_m']:,.0f}",
            )
        console.print(shot_table)

    # Зберігаємо GeoJSON
    if output:
        geojson = result.to_geojson()
        output.write_text(json.dumps(geojson, indent=2))
        console.print(f"\n[green]✅ Траєкторія збережена:[/] {output}")


@sim_app.command("fire")
def sim_fire(
    lat: float = typer.Option(..., "--lat", help="Широта джерела займання"),
    lon: float = typer.Option(..., "--lon", help="Довгота джерела займання"),
    duration: float = typer.Option(6.0, "--duration", "-d", help="Тривалість симуляції (години)", min=0.1),
    radius: float = typer.Option(10.0, "--radius", "-r", help="Радіус симуляційного поля (км)", min=1.0),
    wind_speed: float = typer.Option(3.0, "--wind", help="Швидкість вітру (м/с)"),
    wind_dir: float = typer.Option(270.0, "--wind-dir", help="Напрямок вітру (°)"),
    moisture: float = typer.Option(0.06, "--moisture", help="Вологість палива [0-1]"),
    temperature: float = typer.Option(25.0, "--temp", help="Температура повітря (°C)"),
    humidity: float = typer.Option(40.0, "--humidity", help="Відносна вологість (%)"),
    dem: Optional[Path] = typer.Option(None, "--dem", help="DEM файл для рельєфу"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Вихідний GeoJSON"),
    frames: int = typer.Option(12, "--frames", help="Кількість анімаційних кадрів"),
) -> None:
    """
    Симуляція поширення лісового вогню.

    \b
    Приклад:
        geoengine sim fire --lat 48.5 --lon 24.2 --duration 6 --wind 5
        geoengine sim fire --lat 48.5 --lon 24.2 --duration 12 --dem data/dem.tif -o fire.geojson
    """
    from geoengine.simulation.fire import FireSimulation
    from geoengine.simulation.ballistics import WindVector

    console.print(Panel(
        f"[cyan]Джерело:[/]    {lat:.4f}°N, {lon:.4f}°E\n"
        f"[cyan]Тривалість:[/] {duration} год\n"
        f"[cyan]Радіус:[/]     {radius} км\n"
        f"[cyan]Вітер:[/]      {wind_speed} м/с @ {wind_dir}°\n"
        f"[cyan]Вологість:[/]  {moisture*100:.0f}% палива, {humidity:.0f}% повітря\n"
        f"[cyan]Температура:[/]{temperature}°C",
        title="[bold]🔥 Симуляція Вогню[/]",
        border_style="red",
    ))

    dem_tile = None
    if dem and dem.exists():
        from geoengine.dem.loader import DEMLoader
        dem_tile = DEMLoader().load(dem)
        console.print(f"[dim]  DEM: {dem.name}[/]")

    wind = WindVector(speed_ms=wind_speed, direction_deg=wind_dir)
    sim  = FireSimulation(dem_tile=dem_tile, wind=wind)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Симуляція...")
        result = sim.run(
            ignition_lat=lat,
            ignition_lon=lon,
            duration_hours=duration,
            radius_km=radius,
            moisture=moisture,
            temperature_c=temperature,
            humidity_pct=humidity,
        )
        p.update(task, description="Готово!")

    console.print(
        f"\n[bold]📊 Результат симуляції:[/]\n"
        f"  Площа спалена:  [red]{result.area_burned_ha:.1f} га[/]\n"
        f"  Клітинок:       {result.burned_cells:,} з {result.total_cells:,}\n"
        f"  Відсоток:       {result.burned_fraction*100:.1f}%\n"
        f"  Розмір клітинки:{result.cell_size_m:.0f} м"
    )

    if output:
        # Зберігаємо фінальний периметр + анімаційні кадри
        data = result.to_dict()
        if frames > 0:
            data["frames"] = result.animation_frames(n_frames=frames)
        output.write_text(json.dumps(data, indent=None))  # compact JSON
        console.print(f"\n[green]✅ Результат збережено:[/] {output} ({output.stat().st_size/1024:.0f} КБ)")


# ════════════════════════════════════════════════════════════════
# SERVER КОМАНДИ
# ════════════════════════════════════════════════════════════════

@server_app.command("start")
def server_start(
    host: str = typer.Option("0.0.0.0", "--host", help="Host"),
    port: int = typer.Option(8000, "--port", "-p", help="Port", min=1, max=65535),
    reload: bool = typer.Option(False, "--reload", "-r", help="Автоперезавантаження (dev)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Кількість uvicorn workers"),
    debug: bool = typer.Option(False, "--debug", help="Debug режим (Swagger UI)"),
    log_level: str = typer.Option("info", "--log-level", help="Рівень логів: debug|info|warning"),
) -> None:
    """
    Запустити GeoEngine FastAPI сервер.

    \b
    Приклад:
        geoengine server start
        geoengine server start --port 8001 --reload --debug
        geoengine server start --workers 4 --host 127.0.0.1
    """
    import os
    os.environ["GEOENGINE_DEBUG"] = str(debug).lower()

    console.print(Panel(
        f"[cyan]Host:[/]      {host}:{port}\n"
        f"[cyan]Workers:[/]   {workers}\n"
        f"[cyan]Reload:[/]    {reload}\n"
        f"[cyan]Debug:[/]     {debug}\n"
        f"[cyan]Log:[/]       {log_level}\n\n"
        f"[dim]API docs: http://{host}:{port}/docs[/]" if debug else "",
        title="[bold]🚀 GeoEngine Server[/]",
        border_style="green",
    ))

    try:
        import uvicorn
        uvicorn.run(
            "apps.server.src.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=log_level,
        )
    except ImportError:
        err.print("[red]❌ uvicorn не встановлено. Запусти: uv pip install uvicorn[standard][/]")
        raise typer.Exit(1)


@server_app.command("check")
def server_check(
    url: str = typer.Option("http://localhost:8000", "--url", help="URL сервера"),
) -> None:
    """Перевірити чи запущений сервер."""
    import httpx

    try:
        response = httpx.get(f"{url}/health", timeout=5.0)
        data = response.json()
        console.print(
            f"[green]✅ Сервер запущений[/]\n"
            f"   Status:      {data.get('status', '?')}\n"
            f"   Version:     {data.get('version', '?')}\n"
            f"   Connections: {data.get('connections', 0)}"
        )
    except Exception as exc:
        err.print(f"[red]❌ Сервер недоступний: {exc}[/]")
        raise typer.Exit(1)


# ════════════════════════════════════════════════════════════════
# SEED / QUICK START
# ════════════════════════════════════════════════════════════════

@app.command("seed")
def seed(
    area: str = typer.Option(
        "ukraine",
        "--area", "-a",
        help="Область: ukraine | carpathians | kyiv | custom",
    ),
    zoom: int = typer.Option(9, "--zoom", "-z", help="Zoom рівень", min=6, max=12),
    output_dir: Path = typer.Option(
        Path("data"),
        "--output", "-o",
        help="Вихідна директорія",
    ),
    source: str = typer.Option("terrarium", "--source", "-s"),
) -> None:
    """
    Завантажити початкові тестові дані для запуску.

    \b
    Приклад:
        geoengine seed
        geoengine seed --area carpathians --zoom 10
        geoengine seed --area kyiv --zoom 11 --output data/kyiv
    """
    AREAS: dict[str, tuple[float, float, float, float]] = {
        "ukraine":     (22.0, 44.0, 40.5, 52.5),
        "carpathians": (22.0, 47.5, 26.5, 49.5),
        "kyiv":        (29.8, 50.1, 31.2, 50.9),
        "crimea":      (33.0, 44.3, 36.7, 46.3),
        "donbas":      (36.5, 47.5, 40.5, 49.5),
        "custom":      (24.0, 48.0, 25.0, 49.0),
    }

    if area not in AREAS:
        err.print(f"[red]❌ Невідома область: {area!r}[/]")
        err.print(f"Доступні: {', '.join(AREAS.keys())}")
        raise typer.Exit(1)

    west, south, east, north = AREAS[area]
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[cyan]Область:[/]  {area}\n"
        f"[cyan]BBox:[/]     {west}°, {south}° → {east}°, {north}°\n"
        f"[cyan]Zoom:[/]     {zoom}\n"
        f"[cyan]Джерело:[/]  {source}\n"
        f"[cyan]Директорія:[/]{output_dir}",
        title="[bold]🌱 Seed Data[/]",
        border_style="green",
    ))

    # Рахуємо тайли
    from geoengine.geo.bbox import BBox
    from geoengine.geo.projection import bbox_to_tiles, tile_count_for_bbox
    bbox_obj = BBox(west=west, south=south, east=east, north=north)
    n_tiles  = tile_count_for_bbox(bbox_obj, zoom)

    console.print(f"[dim]  Тайлів: {n_tiles} @ zoom={zoom}[/]")

    if n_tiles > 500:
        console.print(f"[yellow]⚠️  {n_tiles} тайлів — може зайняти час...[/]")

    async def _seed() -> None:
        from geoengine.dem.sources  import DEMSourceManager, DEMSourceID
        from geoengine.dem.processor import merge_tiles, fill_gaps
        from geoengine.io.geotiff   import write_geotiff

        mgr = DEMSourceManager(cache_dir=output_dir / "cache")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Завантаження {n_tiles} тайлів...", total=n_tiles)

            tiles = await mgr.fetch_tiles(
                bbox=bbox_obj,
                zoom=zoom,
                source=DEMSourceID(source),
            )
            progress.update(task, completed=len(tiles))

            if not tiles:
                err.print("[red]❌ Не вдалося завантажити дані[/]")
                return

            progress.update(task, description="Злиття...")
            merged = tiles[0] if len(tiles) == 1 else merge_tiles(tiles)
            if merged.has_nodata:
                merged = fill_gaps(merged)

            out_path = output_dir / f"{area}_z{zoom}.tif"
            progress.update(task, description="Збереження GeoTIFF...")
            write_geotiff(merged, out_path)

        console.print(
            f"\n[green]✅ Дані готові:[/]\n"
            f"   Файл:    {out_path}\n"
            f"   Розмір:  {merged.width}×{merged.height}px\n"
            f"   Висоти:  {merged.min_elevation:.0f}м — {merged.max_elevation:.0f}м\n"
            f"   Тайлів:  {len(tiles)} завантажено\n\n"
            f"[bold]Наступний крок:[/]\n"
            f"   [cyan]geoengine dem info {out_path}[/]\n"
            f"   [cyan]geoengine mesh build {out_path} -o terrain.glb[/]"
        )

    asyncio.run(_seed())


# ════════════════════════════════════════════════════════════════
# УТИЛІТИ
# ════════════════════════════════════════════════════════════════

@app.command("bbox")
def bbox_info(
    coords: str = typer.Argument(
        ...,
        help="BBox у форматі west,south,east,north або назва: ukraine | kyiv | ...",
    ),
) -> None:
    """
    Інформація про BBox та кількість тайлів на різних zoom рівнях.

    \b
    Приклад:
        geoengine bbox ukraine
        geoengine bbox 22,47,26,50
    """
    NAMED = {
        "ukraine":     "22.0,44.0,40.5,52.5",
        "carpathians": "22.0,47.5,26.5,49.5",
        "kyiv":        "29.8,50.1,31.2,50.9",
    }
    if coords in NAMED:
        coords = NAMED[coords]

    try:
        parts = [float(x) for x in coords.split(",")]
        if len(parts) != 4:
            raise ValueError
        west, south, east, north = parts
    except ValueError:
        err.print("[red]❌ Невірний формат. Використовуй: west,south,east,north[/]")
        raise typer.Exit(1)

    from geoengine.geo.bbox import BBox
    from geoengine.geo.projection import tile_count_for_bbox

    bbox = BBox(west=west, south=south, east=east, north=north)

    table = Table(title="📦 BBox Info", border_style="blue")
    table.add_column("Параметр",  style="cyan")
    table.add_column("Значення",  style="white")

    table.add_row("West",    f"{bbox.west:.4f}°")
    table.add_row("South",   f"{bbox.south:.4f}°")
    table.add_row("East",    f"{bbox.east:.4f}°")
    table.add_row("North",   f"{bbox.north:.4f}°")
    table.add_row("Ширина",  f"{bbox.width:.4f}°")
    table.add_row("Висота",  f"{bbox.height:.4f}°")
    table.add_row("Площа",   f"{bbox.area_m2 / 1_000_000:.0f} км²")

    console.print(table)

    zoom_table = Table(title="🔍 Тайли по zoom рівнях", border_style="dim")
    zoom_table.add_column("Zoom", style="cyan", justify="right")
    zoom_table.add_column("Тайлів", style="white", justify="right")
    zoom_table.add_column("Роздільність", style="dim")

    resolutions = {
        4: "~4900 м/пікс",
        6: "~1200 м/пікс",
        8: "~300 м/пікс",
        9: "~150 м/пікс",
        10: "~75 м/пікс",
        11: "~38 м/пікс",
        12: "~19 м/пікс",
        13: "~9.5 м/пікс",
        14: "~4.8 м/пікс",
    }
    for z in range(4, 15):
        n = tile_count_for_bbox(bbox, z)
        res = resolutions.get(z, "")
        color = "green" if n <= 100 else "yellow" if n <= 500 else "red"
        zoom_table.add_row(str(z), f"[{color}]{n:,}[/]", res)

    console.print(zoom_table)


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()
