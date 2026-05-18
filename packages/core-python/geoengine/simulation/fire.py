"""
GeoEngine — Fire Spread Simulation
Поширення лісового вогню з урахуванням рельєфу та вітру.

Модель: Rothermel (спрощена) + Cellular Automata
  - Швидкість розповсюдження залежить від:
    * Схилу (крутий схил → вогонь швидше)
    * Вітру (вітер у бік вогню → прискорення)
    * Типу рослинності (з OSM landuse/natural)
  - Клітинний автомат на регулярній сітці
  - Кожна клітинка: Not Burned / Burning / Burned

Виходи:
  - Часова карта (коли кожна клітинка загорілась)
  - Периметр вогню як GeoJSON
  - Анімація кадрів (для браузера)

Застосування:
  - Управління лісовими пожежами (ДСНС)
  - Планування протипожежних заходів
  - Оцінка ризику для населених пунктів

Usage:
    sim = FireSimulation(dem_tile, wind=WindVector(5, 270))
    result = sim.run(
        ignition_lat=48.5, ignition_lon=24.2,
        duration_hours=6.0,
    )
    geojson = result.perimeter_at_time(3.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from .ballistics import WindVector

log: structlog.BoundLogger = structlog.get_logger(__name__)


# ── КОНСТАНТИ ──────────────────────────────────────────────────

# Типи рослинності та їх горючість (базова швидкість поширення, м/хв)
FUEL_TYPES: dict[str, dict[str, float]] = {
    "forest":      {"spread_rate": 4.0,  "intensity": 3000.0, "moisture": 0.08},
    "wood":        {"spread_rate": 4.0,  "intensity": 3000.0, "moisture": 0.08},
    "grassland":   {"spread_rate": 8.0,  "intensity": 1500.0, "moisture": 0.04},
    "farmland":    {"spread_rate": 3.0,  "intensity": 800.0,  "moisture": 0.10},
    "scrub":       {"spread_rate": 5.0,  "intensity": 2000.0, "moisture": 0.06},
    "heath":       {"spread_rate": 6.0,  "intensity": 2200.0, "moisture": 0.05},
    "residential": {"spread_rate": 1.0,  "intensity": 500.0,  "moisture": 0.15},
    "bare_rock":   {"spread_rate": 0.0,  "intensity": 0.0,    "moisture": 1.0},
    "water":       {"spread_rate": 0.0,  "intensity": 0.0,    "moisture": 1.0},
    "default":     {"spread_rate": 3.0,  "intensity": 1000.0, "moisture": 0.07},
}

# Вологість палива — коефіцієнт до spread_rate
def moisture_factor(moisture: float) -> float:
    """Вплив вологості на швидкість поширення."""
    return max(0.0, 1.0 - moisture * 5.0)


class CellState(IntEnum):
    NOT_BURNED = 0
    BURNING    = 1
    BURNED     = 2
    FIREBREAK  = 3    # непрохідна перешкода (вода, скеля)


@dataclass(slots=True)
class FireCell:
    """Одна клітинка сітки симуляції."""
    state:        CellState = CellState.NOT_BURNED
    ignite_time:  float = float("inf")  # час займання, хвилини
    burn_out_time: float = float("inf") # час догоряння, хвилини
    fuel_type:    str   = "default"
    elevation:    float = 0.0
    slope_deg:    float = 0.0
    aspect_deg:   float = 0.0


@dataclass
class FireResult:
    """Результат симуляції."""

    # Карта часу займання (хвилини, inf = не горіло)
    ignition_time_map: npt.NDArray[np.float32]

    # Розмір клітинки, м
    cell_size_m: float

    # BBox симуляції
    west:  float
    south: float
    east:  float
    north: float

    # Метрики
    total_cells:      int
    burned_cells:     int
    duration_hours:   float
    area_burned_ha:   float

    # Позиція джерела
    ignition_lat: float
    ignition_lon: float

    @property
    def burned_fraction(self) -> float:
        return self.burned_cells / max(self.total_cells, 1)

    def perimeter_at_time(self, hours: float) -> dict:
        """
        GeoJSON периметр вогню на момент часу.

        Args:
            hours: час від початку симуляції (години)

        Returns:
            GeoJSON Polygon або MultiPolygon
        """
        minutes = hours * 60.0
        burned  = self.ignition_time_map <= minutes

        return _array_to_perimeter_geojson(
            burned,
            west=self.west, south=self.south,
            east=self.east, north=self.north,
            hours=hours,
        )

    def animation_frames(
        self,
        n_frames: int = 24,
    ) -> list[dict]:
        """
        Список GeoJSON кадрів для анімації.

        Returns:
            Список dict {time_hours, geojson, burned_fraction}
        """
        frames   = []
        max_time = self.duration_hours
        for i in range(n_frames + 1):
            t_hours   = max_time * i / n_frames
            t_minutes = t_hours * 60.0
            burned    = self.ignition_time_map <= t_minutes
            n_burned  = int(burned.sum())
            fraction  = n_burned / max(self.total_cells, 1)

            frames.append({
                "time_hours":     round(t_hours, 2),
                "burned_cells":   n_burned,
                "burned_fraction": round(fraction, 4),
                "area_ha":        round(n_burned * self.cell_size_m**2 / 10000, 1),
                "geojson":        _array_to_perimeter_geojson(
                    burned,
                    self.west, self.south, self.east, self.north,
                    t_hours,
                ),
            })
        return frames

    def to_dict(self) -> dict:
        """Серіалізація для REST/WS."""
        return {
            "type":            "fire_result",
            "duration_hours":  self.duration_hours,
            "area_burned_ha":  round(self.area_burned_ha, 1),
            "burned_fraction": round(self.burned_fraction, 4),
            "burned_cells":    self.burned_cells,
            "total_cells":     self.total_cells,
            "cell_size_m":     self.cell_size_m,
            "bbox": {
                "west": self.west, "south": self.south,
                "east": self.east, "north": self.north,
            },
            "ignition": {
                "lat": self.ignition_lat,
                "lon": self.ignition_lon,
            },
            # base64 карта часів займання
            "ignition_time_map": _encode_f32(self.ignition_time_map),
            "map_width":  self.ignition_time_map.shape[1],
            "map_height": self.ignition_time_map.shape[0],
        }


# ── FIRE SIMULATION ────────────────────────────────────────────

class FireSimulation:
    """
    Клітинний автомат для моделювання поширення вогню.

    Алгоритм:
      1. Ініціалізувати сітку з DEM висотами та типами рослинності
      2. Запустити клітинку-джерело у момент t=0
      3. Кожен крок: для кожної горячої клітинки
         перевіряємо 8 сусідів
         → обчислюємо швидкість поширення (Rothermel)
         → якщо ймовірність > поріг → займаємо сусіда
      4. Зберігаємо час займання кожної клітинки

    Параметри сітки:
      Розмір клітинки: cell_size_m (30-100м для DEM точності)
      Сітка будується з BBox навколо джерела
    """

    def __init__(
        self,
        dem_tile:     Any | None = None,    # DEMTile
        wind:         WindVector | None = None,
        fuel_map:     dict | None = None,   # {(row,col): fuel_type}
        cell_size_m:  float = 50.0,         # розмір клітинки
        dt_min:       float = 1.0,          # часовий крок симуляції (хвилини)
    ) -> None:
        self._dem       = dem_tile
        self._wind      = wind or WindVector()
        self._fuel_map  = fuel_map or {}
        self._cell_size = cell_size_m
        self._dt        = dt_min

    def run(
        self,
        ignition_lat:   float,
        ignition_lon:   float,
        duration_hours: float = 6.0,
        radius_km:      float = 20.0,       # радіус симуляційної сітки
        moisture:       float = 0.06,       # початкова вологість
        temperature_c:  float = 25.0,       # температура повітря
        humidity_pct:   float = 40.0,       # відносна вологість, %
    ) -> FireResult:
        """
        Запустити симуляцію.

        Args:
            ignition_lat:   широта джерела займання
            ignition_lon:   довгота джерела займання
            duration_hours: тривалість симуляції (години)
            radius_km:      радіус симуляційного поля (км)
            moisture:       вологість палива [0..1]
            temperature_c:  температура (°C)
            humidity_pct:   відносна вологість (%)

        Returns:
            FireResult
        """
        log.info(
            "fire.start",
            lat=ignition_lat, lon=ignition_lon,
            duration_h=duration_hours,
            radius_km=radius_km,
        )

        # Побудувати сітку
        grid, meta = self._build_grid(
            ignition_lat, ignition_lon,
            radius_km, moisture,
        )
        rows, cols = grid.shape[:2]

        # Вологість залежна від умов
        effective_moisture = self._calc_effective_moisture(
            moisture, temperature_c, humidity_pct
        )

        # Знайти клітинку-джерело
        ig_row, ig_col = meta["ignition_row"], meta["ignition_col"]
        if not (0 <= ig_row < rows and 0 <= ig_col < cols):
            raise ValueError("Точка займання поза межами сітки")

        # Карта часів займання (inf = не горіло)
        ignition_map = np.full((rows, cols), np.inf, dtype=np.float32)
        ignition_map[ig_row, ig_col] = 0.0

        # Стан клітинок: 0=не горіло, 1=горить, 2=згоріло, 3=firebreak
        state = grid[:, :, 0].copy()    # fuel type encoded as 0/3

        # Список горячих клітинок (row, col)
        burning: list[tuple[int, int]] = [(ig_row, ig_col)]
        state[ig_row, ig_col] = CellState.BURNING

        # Час догоряння (хвилини) — горить ~30-60 хв
        burn_duration = np.full((rows, cols), 45.0, dtype=np.float32)

        t_min = 0.0
        max_time_min = duration_hours * 60.0

        # ── Головний цикл ─────────────────────────────────────
        while burning and t_min < max_time_min:
            new_burning: list[tuple[int, int]] = []

            for r, c in burning:
                # Перевіряємо 8 сусідів
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if state[nr, nc] != CellState.NOT_BURNED:
                            continue

                        # Швидкість поширення (м/хв)
                        spread = self._spread_rate(
                            r, c, nr, nc,
                            grid, effective_moisture,
                            dr, dc,
                        )
                        if spread <= 0.0:
                            continue

                        # Час досягнення сусіда
                        dist_m = self._cell_size * (1.0 if (dr == 0 or dc == 0) else math.sqrt(2))
                        dt_reach = dist_m / spread

                        ignite_t = ignition_map[r, c] + dt_reach
                        if ignite_t < ignition_map[nr, nc]:
                            ignition_map[nr, nc] = ignite_t

                        if ignite_t <= t_min + self._dt:
                            state[nr, nc] = CellState.BURNING
                            new_burning.append((nr, nc))

                # Клітинка догоріла?
                if t_min - ignition_map[r, c] >= burn_duration[r, c]:
                    state[r, c] = CellState.BURNED

            burning = [
                (r, c) for r, c in burning
                if state[r, c] == CellState.BURNING
            ] + new_burning

            t_min += self._dt

        # ── Метрики ──────────────────────────────────────────
        burned_count  = int((ignition_map < np.inf).sum())
        total_count   = rows * cols
        area_ha       = burned_count * self._cell_size**2 / 10_000

        log.info(
            "fire.done",
            burned_cells=burned_count,
            area_ha=round(area_ha, 1),
            duration_h=duration_hours,
        )

        return FireResult(
            ignition_time_map=ignition_map,
            cell_size_m=self._cell_size,
            west=meta["west"], south=meta["south"],
            east=meta["east"], north=meta["north"],
            total_cells=total_count,
            burned_cells=burned_count,
            duration_hours=duration_hours,
            area_burned_ha=area_ha,
            ignition_lat=ignition_lat,
            ignition_lon=ignition_lon,
        )

    # ── Приватні методи ──────────────────────────────────────

    def _build_grid(
        self,
        lat: float, lon: float,
        radius_km: float,
        moisture:  float,
    ) -> tuple[npt.NDArray, dict]:
        """
        Побудувати сітку симуляції навколо точки.

        Повертає:
          grid: (rows, cols, 2) де [:,:,0] = firebreak_mask, [:,:,1] = slope
          meta: dict з bbox та індексом ignition
        """
        R      = 6_371_000.0
        cosLat = math.cos(math.radians(lat))

        # Розмір поля
        d_lat  = math.degrees(radius_km * 1000 / R)
        d_lon  = math.degrees(radius_km * 1000 / (R * cosLat))

        west   = lon - d_lon
        east   = lon + d_lon
        south  = lat - d_lat
        north  = lat + d_lat

        # Розміри сітки
        cols   = max(10, int((east  - west)  * 111_320 * cosLat / self._cell_size))
        rows   = max(10, int((north - south) * 111_320            / self._cell_size))

        # Обмежуємо розмір
        cols = min(cols, 500)
        rows = min(rows, 500)

        # Позиція джерела у сітці
        ig_col = int((lon - west)  / (east  - west)  * cols)
        ig_row = int((lat - south) / (north - south)  * rows)
        ig_col = max(0, min(cols - 1, ig_col))
        ig_row = max(0, min(rows - 1, ig_row))

        # Ініціалізувати масив
        # [:,:,0] = state (0=горюче, 3=firebreak)
        # [:,:,1] = slope (градуси)
        grid = np.zeros((rows, cols, 2), dtype=np.float32)

        # Заповнити slope з DEM
        if self._dem is not None:
            for r in range(rows):
                for c in range(cols):
                    cell_lat = south + (r + 0.5) / rows * (north - south)
                    cell_lon = west  + (c + 0.5) / cols * (east  - west)
                    h = self._dem.sample(cell_lat, cell_lon)
                    if h is None:
                        grid[r, c, 0] = CellState.FIREBREAK
                    # slope буде обчислено окремо

            # Обчислити slope
            from geoengine.dem.analysis import compute_slope
            from geoengine.dem.loader   import DEMTile
            slope_result = compute_slope(self._dem)
            # Rescale slope map до нашої сітки
            from scipy.ndimage import zoom as scipy_zoom
            scale_r = rows / slope_result.degrees.shape[0]
            scale_c = cols / slope_result.degrees.shape[1]
            slope_resized = scipy_zoom(
                slope_result.degrees, (scale_r, scale_c), order=1
            )
            grid[:, :, 1] = slope_resized[:rows, :cols]

        return grid, {
            "west": west, "south": south,
            "east": east, "north": north,
            "ignition_row": ig_row,
            "ignition_col": ig_col,
        }

    def _spread_rate(
        self,
        src_r: int, src_c: int,
        dst_r: int, dst_c: int,
        grid:  npt.NDArray,
        moisture: float,
        dr: int, dc: int,
    ) -> float:
        """
        Швидкість поширення від src до dst (м/хв).
        Rothermel спрощена формула.
        """
        # Firebreak перевірка
        if grid[dst_r, dst_c, 0] == CellState.FIREBREAK:
            return 0.0

        # Базова швидкість за типом палива
        fuel_key = self._fuel_map.get((dst_r, dst_c), "default")
        fuel = FUEL_TYPES.get(fuel_key, FUEL_TYPES["default"])
        base_rate = fuel["spread_rate"]

        if base_rate <= 0.0:
            return 0.0

        # Вологість
        base_rate *= moisture_factor(moisture)

        # Вплив схилу (Rothermel: exp(0.069 × slope))
        slope_deg = float(grid[dst_r, dst_c, 1])
        # Визначаємо чи схил сприяє або протидіє поширенню
        # (спрощено: порівнюємо висоти)
        slope_src = float(grid[src_r, src_c, 1])
        slope_factor = 1.0
        if slope_deg > 0:
            # Якщо поширення вгору по схилу — прискорення
            # (спрощена версія без точного напрямку схилу)
            slope_factor = math.exp(0.069 * min(slope_deg, 45.0))

        # Вплив вітру
        wind_factor = self._wind_factor(dr, dc)

        return base_rate * slope_factor * wind_factor

    def _wind_factor(self, dr: int, dc: int) -> float:
        """
        Коефіцієнт вітру для напрямку поширення (dr, dc).
        Rothermel: C × U^B де U = швидкість вітру (mph).
        Спрощена версія.
        """
        speed = self._wind.speed_ms
        if speed < 0.1:
            return 1.0

        # Напрямок поширення (кут від Півдня)
        spread_az = math.degrees(math.atan2(dc, dr)) % 360

        # Напрямок вітру (куди дме = direction + 180)
        wind_toward = (self._wind.direction_deg + 180) % 360

        # Кут між напрямком вітру та поширенням
        angle_diff = abs(spread_az - wind_toward) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # cos(angle): 0° = максимальне підсилення, 180° = затримка
        cos_a = math.cos(math.radians(angle_diff))

        # Емпіричний коефіцієнт
        speed_mph = speed * 2.237
        if cos_a > 0:
            return 1.0 + 0.15 * cos_a * speed_mph
        else:
            return max(0.1, 1.0 + 0.05 * cos_a * speed_mph)

    @staticmethod
    def _calc_effective_moisture(
        base_moisture: float,
        temperature_c: float,
        humidity_pct:  float,
    ) -> float:
        """
        Ефективна вологість з урахуванням метеоумов.
        Вища температура та нижча вологість → сухіше паливо.
        """
        temp_factor = max(0.5, 1.0 - (temperature_c - 20.0) * 0.01)
        humid_factor = humidity_pct / 100.0
        return base_moisture * temp_factor * (0.5 + 0.5 * humid_factor)


# ── ДОПОМІЖНІ ФУНКЦІЇ ──────────────────────────────────────────

def _array_to_perimeter_geojson(
    burned:    npt.NDArray[np.bool_],
    west:      float,
    south:     float,
    east:      float,
    north:     float,
    hours:     float,
) -> dict:
    """
    Конвертувати бінарний масив згорілих клітинок у GeoJSON.
    Спрощена версія: повертає Polygon з convex hull.
    """
    rows, cols = burned.shape
    points     = []

    for r in range(rows):
        for c in range(cols):
            if burned[r, c]:
                lat = south + (r + 0.5) / rows * (north - south)
                lon = west  + (c + 0.5) / cols * (east  - west)
                points.append((lon, lat))

    if not points:
        return {
            "type": "FeatureCollection",
            "features": [],
        }

    # Простий convex hull (Graham scan)
    hull = _convex_hull(points)

    # Закрити полігон
    if hull[0] != hull[-1]:
        hull.append(hull[0])

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [hull],
            },
            "properties": {
                "time_hours":    round(hours, 2),
                "burned_cells":  int(burned.sum()),
            },
        }],
    }


def _convex_hull(points: list[tuple[float, float]]) -> list[list[float]]:
    """Graham scan convex hull."""
    if len(points) < 3:
        return [[p[0], p[1]] for p in points]

    pts = sorted(set(points))

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower: list[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [[round(p[0], 7), round(p[1], 7)] for p in hull]


def _encode_f32(arr: npt.NDArray) -> str:
    """base64 encode float32 array."""
    import base64
    clean = np.where(np.isinf(arr), float(arr[~np.isinf(arr)].max()*1.1)
                     if arr[~np.isinf(arr)].size > 0 else 0.0, arr)
    return base64.b64encode(
        clean.astype(np.float32).tobytes()
    ).decode("ascii")
