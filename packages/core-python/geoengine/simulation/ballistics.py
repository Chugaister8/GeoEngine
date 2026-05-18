"""
GeoEngine — Ballistics Simulation
Траєкторія снаряду з урахуванням реального рельєфу.

Фізична модель:
  - Гравітація (9.81 м/с²)
  - Опір повітря (drag coefficient)
  - Вітер (вектор 3D)
  - Ефект Коріоліса (для великих відстаней)
  - Перевірка перетину з DEM (terrain hit detection)

Координатна система:
  ENU відносно точки пострілу (East, North, Up)
  Висоти беруться з DEMTile.sample()

Застосування:
  - Артилерія (непряма наводка)
  - Авіабомби
  - Ракети (без корекції)
  - Планування маршрутів БПЛА

Usage:
    solver = BallisticsSolver(dem_tile)
    result = solver.solve(
        origin=LatLonAlt(48.5, 24.2, 500),
        azimuth_deg=45.0,
        elevation_deg=30.0,
        muzzle_velocity=800.0,
    )
    print(result.impact_point)
    print(result.max_range_m)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ── Фізичні константи ──────────────────────────────────────────
G          = 9.80665      # прискорення вільного падіння, м/с²
RHO_0      = 1.225        # густота повітря на рівні моря, кг/м³
SCALE_H    = 8500.0       # висотна шкала атмосфери, м
OMEGA_EARTH = 7.2921e-5   # кутова швидкість Землі, рад/с
R_EARTH    = 6_371_000.0  # радіус Землі, м

# ── Типові балістичні коефіцієнти ──────────────────────────────
# BC = m / (Cd × A)  [кг/м²]
BALLISTIC_PRESETS: dict[str, dict[str, float]] = {
    "artillery_152mm": {
        "mass_kg":   43.5,
        "diameter_m": 0.152,
        "cd":         0.30,    # drag coefficient
        "bc":         0.420,
    },
    "artillery_122mm": {
        "mass_kg":   21.8,
        "diameter_m": 0.122,
        "cd":         0.28,
        "bc":         0.388,
    },
    "mortar_120mm": {
        "mass_kg":   15.9,
        "diameter_m": 0.120,
        "cd":         0.40,
        "bc":         0.220,
    },
    "rifle_762x54": {
        "mass_kg":   0.0098,
        "diameter_m": 0.00762,
        "cd":         0.295,
        "bc":         0.387,
    },
    "bomb_250kg": {
        "mass_kg":   250.0,
        "diameter_m": 0.38,
        "cd":         0.50,
        "bc":         0.580,
    },
}


# ── Типи даних ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LatLonAlt:
    """Географічна точка з висотою."""
    lat: float
    lon: float
    alt: float = 0.0


@dataclass(slots=True)
class ProjectileParams:
    """
    Параметри снаряду.
    BC (Ballistic Coefficient) = m / (Cd × A)
    """
    mass_kg:    float           # маса, кг
    diameter_m: float           # діаметр, м
    cd:         float = 0.30    # коефіцієнт лобового опору
    bc:         float = 0.0     # балістичний коефіцієнт (якщо 0 — обчислюється)

    def __post_init__(self) -> None:
        if self.bc <= 0.0 and self.diameter_m > 0:
            area    = math.pi * (self.diameter_m / 2) ** 2
            self.bc = self.mass_kg / (self.cd * area)

    @property
    def cross_section_m2(self) -> float:
        return math.pi * (self.diameter_m / 2) ** 2

    @classmethod
    def from_preset(cls, name: str) -> "ProjectileParams":
        if name not in BALLISTIC_PRESETS:
            raise ValueError(
                f"Невідомий пресет: {name!r}. "
                f"Доступні: {list(BALLISTIC_PRESETS.keys())}"
            )
        return cls(**BALLISTIC_PRESETS[name])


@dataclass(slots=True)
class WindVector:
    """Вектор вітру."""
    speed_ms:   float = 0.0    # швидкість, м/с
    direction_deg: float = 0.0 # звідки дме (0=Північ, 90=Схід)
    vertical_ms: float = 0.0   # вертикальна складова

    @property
    def east_ms(self) -> float:
        return self.speed_ms * math.sin(math.radians(self.direction_deg))

    @property
    def north_ms(self) -> float:
        return self.speed_ms * math.cos(math.radians(self.direction_deg))


@dataclass
class BallisticsResult:
    """Результат балістичного розрахунку."""

    # Точка падіння
    impact_point:     LatLonAlt | None

    # Траєкторія (ENU відносно origin)
    trajectory_east:  npt.NDArray[np.float64]
    trajectory_north: npt.NDArray[np.float64]
    trajectory_up:    npt.NDArray[np.float64]
    trajectory_time:  npt.NDArray[np.float64]

    # Метрики
    max_height_m:    float   # максимальна висота (над точкою пострілу)
    max_range_m:     float   # горизонтальна відстань до падіння
    flight_time_s:   float   # час польоту
    impact_velocity: float   # швидкість у момент удару, м/с
    impact_angle_deg: float  # кут падіння (від горизонталі)

    # Статус
    hit_terrain:     bool    # чи вдарився у рельєф
    hit_bbox:        bool    # чи вийшов за межі DEM

    @property
    def trajectory_latlon(self) -> list[LatLonAlt]:
        """Траєкторія у географічних координатах."""
        return _enu_to_latlon(
            self._origin_lat,
            self._origin_lon,
            self._origin_alt,
            self.trajectory_east,
            self.trajectory_north,
            self.trajectory_up,
        )

    # Internal (заповнюється solver'ом)
    _origin_lat: float = field(default=0.0, repr=False)
    _origin_lon: float = field(default=0.0, repr=False)
    _origin_alt: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict:
        """Серіалізація для REST/WS."""
        traj = []
        for e, n, u, t in zip(
            self.trajectory_east, self.trajectory_north,
            self.trajectory_up, self.trajectory_time,
            strict=True,
        ):
            traj.append({
                "east": round(float(e), 1),
                "north": round(float(n), 1),
                "up": round(float(u), 1),
                "time": round(float(t), 3),
            })

        impact = None
        if self.impact_point:
            impact = {
                "lat": round(self.impact_point.lat, 7),
                "lon": round(self.impact_point.lon, 7),
                "alt": round(self.impact_point.alt, 1),
            }

        return {
            "type":            "ballistics_result",
            "impact":          impact,
            "trajectory":      traj,
            "max_height_m":    round(self.max_height_m, 1),
            "max_range_m":     round(self.max_range_m, 1),
            "flight_time_s":   round(self.flight_time_s, 3),
            "impact_velocity": round(self.impact_velocity, 1),
            "impact_angle_deg": round(self.impact_angle_deg, 1),
            "hit_terrain":     self.hit_terrain,
            "hit_bbox":        self.hit_bbox,
        }

    def to_geojson(self) -> dict:
        """Траєкторія як GeoJSON LineString."""
        coords = [
            [self._origin_lon, self._origin_lat, self._origin_alt]
        ]
        R      = R_EARTH
        cosLat = math.cos(math.radians(self._origin_lat))

        for e, n, u in zip(
            self.trajectory_east,
            self.trajectory_north,
            self.trajectory_up,
            strict=True,
        ):
            lat = self._origin_lat + math.degrees(n / R)
            lon = self._origin_lon + math.degrees(e / (R * cosLat))
            alt = self._origin_alt + u
            coords.append([round(lon, 7), round(lat, 7), round(alt, 1)])

        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "max_height_m":  self.max_height_m,
                        "max_range_m":   self.max_range_m,
                        "flight_time_s": self.flight_time_s,
                        "hit_terrain":   self.hit_terrain,
                    },
                }
            ],
        }


# ── BALLISTICS SOLVER ──────────────────────────────────────────

class BallisticsSolver:
    """
    Чисельне рішення рівнянь руху снаряду.

    Метод інтегрування: Runge-Kutta 4 (RK4)
    Часовий крок: адаптивний (менший при малій висоті)

    Перевірка зіткнення з рельєфом:
      Після кожного кроку порівнюємо висоту снаряду
      з висотою рельєфу в цій точці (DEM.sample).
      Якщо снаряд нижче рельєфу → лінійна інтерполяція точки зіткнення.

    Coordinate system:
      ENU відносно launch_point:
        east  = X (метри на схід)
        north = Y (метри на північ)
        up    = Z (метри вгору)
    """

    def __init__(
        self,
        dem_tile:       Any | None = None,    # DEMTile
        max_time_s:     float = 300.0,        # максимальний час польоту
        dt_base:        float = 0.05,         # базовий часовий крок, с
        store_interval: float = 0.5,          # інтервал запису траєкторії, с
        coriolis:       bool  = False,        # ефект Коріоліса
    ) -> None:
        self._dem           = dem_tile
        self._max_time      = max_time_s
        self._dt            = dt_base
        self._store_interval = store_interval
        self._coriolis      = coriolis

    def solve(
        self,
        origin:           LatLonAlt,
        azimuth_deg:      float,
        elevation_deg:    float,
        muzzle_velocity:  float,
        projectile:       ProjectileParams | str = "artillery_122mm",
        wind:             WindVector | None = None,
    ) -> BallisticsResult:
        """
        Розрахувати траєкторію снаряду.

        Args:
            origin:          Точка пострілу (LatLonAlt)
            azimuth_deg:     Азимут пострілу (0=Північ, 90=Схід)
            elevation_deg:   Кут підвищення (градуси, 0=горизонт)
            muzzle_velocity: Початкова швидкість, м/с
            projectile:      ProjectileParams або ім'я пресету
            wind:            Вітер (None = немає)

        Returns:
            BallisticsResult з траєкторією та точкою падіння
        """
        if isinstance(projectile, str):
            projectile = ProjectileParams.from_preset(projectile)

        wind = wind or WindVector()

        log.debug(
            "ballistics.solve",
            azimuth=azimuth_deg,
            elevation=elevation_deg,
            v0=muzzle_velocity,
            projectile_bc=round(projectile.bc, 3),
        )

        # Початкова швидкість у ENU
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)

        cos_el = math.cos(el_rad)
        sin_el = math.sin(el_rad)

        vE0 = muzzle_velocity * cos_el * math.sin(az_rad)
        vN0 = muzzle_velocity * cos_el * math.cos(az_rad)
        vU0 = muzzle_velocity * sin_el

        # Початкова позиція (ENU)
        e0, n0, u0 = 0.0, 0.0, 0.0

        # Висота над рівнем моря в точці пострілу
        origin_asl = origin.alt

        # ── RK4 інтегрування ────────────────────────────────────
        e, n, u   = e0, n0, u0
        vE, vN, vU = vE0, vN0, vU0
        t         = 0.0

        # Зберігаємо траєкторію
        traj_e: list[float] = [e]
        traj_n: list[float] = [n]
        traj_u: list[float] = [u]
        traj_t: list[float] = [t]

        next_store = self._store_interval
        max_h      = 0.0
        impact_point: LatLonAlt | None = None
        hit_terrain = False
        hit_bbox    = False

        prev_e, prev_n, prev_u = e, n, u

        while t < self._max_time:
            # Адаптивний крок: менший поблизу рельєфу
            dt = self._adaptive_dt(u)

            # Висота рельєфу в поточній точці
            terrain_h = self._terrain_height_at(
                e, n, origin, origin_asl
            )
            if terrain_h is None:
                # Вийшли за межі DEM
                hit_bbox = True
                break

            # Перевірка чи знаходимось нижче рельєфу
            current_h = origin_asl + u
            if current_h < terrain_h and t > 0.01:
                # Знайшли зіткнення — інтерполюємо
                impact_point, impact_u = self._interpolate_impact(
                    prev_e, prev_n, prev_u,
                    e, n, u,
                    origin, origin_asl,
                )
                hit_terrain = True
                u = impact_u
                break

            # RK4 крок
            def derivatives(
                _e: float, _n: float, _u: float,
                _vE: float, _vN: float, _vU: float,
            ) -> tuple[float, float, float, float, float, float]:
                return self._equations_of_motion(
                    _e, _n, _u, _vE, _vN, _vU,
                    projectile, wind,
                    origin.lat, t,
                )

            k1 = derivatives(e, n, u, vE, vN, vU)
            k2 = derivatives(
                e  + dt/2*k1[0], n  + dt/2*k1[1],
                u  + dt/2*k1[2],
                vE + dt/2*k1[3], vN + dt/2*k1[4],
                vU + dt/2*k1[5],
            )
            k3 = derivatives(
                e  + dt/2*k2[0], n  + dt/2*k2[1],
                u  + dt/2*k2[2],
                vE + dt/2*k2[3], vN + dt/2*k2[4],
                vU + dt/2*k2[5],
            )
            k4 = derivatives(
                e  + dt*k3[0], n  + dt*k3[1],
                u  + dt*k3[2],
                vE + dt*k3[3], vN + dt*k3[4],
                vU + dt*k3[5],
            )

            prev_e, prev_n, prev_u = e, n, u

            e  += dt/6 * (k1[0]+2*k2[0]+2*k3[0]+k4[0])
            n  += dt/6 * (k1[1]+2*k2[1]+2*k3[1]+k4[1])
            u  += dt/6 * (k1[2]+2*k2[2]+2*k3[2]+k4[2])
            vE += dt/6 * (k1[3]+2*k2[3]+2*k3[3]+k4[3])
            vN += dt/6 * (k1[4]+2*k2[4]+2*k3[4]+k4[4])
            vU += dt/6 * (k1[5]+2*k2[5]+2*k3[5]+k4[5])
            t  += dt

            # Запис траєкторії
            if t >= next_store:
                traj_e.append(e)
                traj_n.append(n)
                traj_u.append(u)
                traj_t.append(t)
                next_store += self._store_interval

            # Максимальна висота
            if u > max_h:
                max_h = u

            # Перевірка чи повернувся до рівня старту (приземлення
            # без рельєфу або при відсутньому DEM)
            if self._dem is None and u < -100.0:
                impact_point = _enu_to_latlon_single(
                    origin.lat, origin.lon, origin.alt,
                    e, n, u,
                )
                break

        # ── Результат ──────────────────────────────────────────
        speed = math.sqrt(vE**2 + vN**2 + vU**2)
        horiz = math.sqrt(vE**2 + vN**2)
        impact_angle = math.degrees(math.atan2(abs(vU), horiz)) if horiz > 0 else 90.0
        max_range    = math.sqrt(e**2 + n**2)

        # Якщо не вдарився у рельєф — кінцева точка
        if impact_point is None:
            impact_point = _enu_to_latlon_single(
                origin.lat, origin.lon, origin.alt,
                e, n, u,
            )

        result = BallisticsResult(
            impact_point=impact_point,
            trajectory_east=np.array(traj_e, dtype=np.float64),
            trajectory_north=np.array(traj_n, dtype=np.float64),
            trajectory_up=np.array(traj_u, dtype=np.float64),
            trajectory_time=np.array(traj_t, dtype=np.float64),
            max_height_m=round(max_h, 2),
            max_range_m=round(max_range, 1),
            flight_time_s=round(t, 3),
            impact_velocity=round(speed, 2),
            impact_angle_deg=round(impact_angle, 2),
            hit_terrain=hit_terrain,
            hit_bbox=hit_bbox,
        )
        result._origin_lat = origin.lat
        result._origin_lon = origin.lon
        result._origin_alt = origin.alt

        log.info(
            "ballistics.result",
            range_m=result.max_range_m,
            time_s=result.flight_time_s,
            max_h=result.max_height_m,
            hit=hit_terrain,
        )
        return result

    def solve_range_table(
        self,
        origin:          LatLonAlt,
        muzzle_velocity: float,
        projectile:      ProjectileParams | str = "artillery_122mm",
        elevations_deg:  list[float] | None = None,
        azimuths_deg:    list[float] | None = None,
        wind:            WindVector | None = None,
    ) -> list[dict]:
        """
        Таблиця стрільби: кілька кутів підвищення → дальності.

        Args:
            elevations_deg: список кутів [5, 10, 15, ...]
            azimuths_deg:   список азимутів (якщо None → тільки 0°)

        Returns:
            Список dict {elevation, azimuth, range_m, flight_time_s, max_height_m}
        """
        if elevations_deg is None:
            elevations_deg = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        if azimuths_deg is None:
            azimuths_deg = [0.0]

        table: list[dict] = []
        for az in azimuths_deg:
            for el in elevations_deg:
                try:
                    r = self.solve(
                        origin=origin,
                        azimuth_deg=az,
                        elevation_deg=el,
                        muzzle_velocity=muzzle_velocity,
                        projectile=projectile,
                        wind=wind,
                    )
                    table.append({
                        "elevation_deg": el,
                        "azimuth_deg":   az,
                        "range_m":       r.max_range_m,
                        "flight_time_s": r.flight_time_s,
                        "max_height_m":  r.max_height_m,
                        "impact_angle_deg": r.impact_angle_deg,
                    })
                except Exception as exc:
                    log.warning(
                        "ballistics.table_error",
                        el=el, az=az, error=str(exc)[:60]
                    )
        return table

    # ── Приватні методи ──────────────────────────────────────

    def _equations_of_motion(
        self,
        e: float, n: float, u: float,
        vE: float, vN: float, vU: float,
        proj:   ProjectileParams,
        wind:   WindVector,
        lat_deg: float,
        t:      float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Похідні для RK4.
        Повертає (dE/dt, dN/dt, dU/dt, dvE/dt, dvN/dt, dvU/dt).
        """
        # Відносна швидкість (снаряд — вітер)
        vrE = vE - wind.east_ms
        vrN = vN - wind.north_ms
        vrU = vU - wind.vertical_ms
        v_rel = math.sqrt(vrE**2 + vrN**2 + vrU**2)

        # Густота повітря на поточній висоті (барометрична формула)
        alt_asl = u + 500.0  # приблизна висота над рівнем моря
        rho = RHO_0 * math.exp(-max(0.0, alt_asl) / SCALE_H)

        # Сила аеродинамічного опору
        if v_rel > 0.001:
            area = proj.cross_section_m2
            Fd   = 0.5 * rho * proj.cd * area * v_rel**2
            aD_E = -(Fd / proj.mass_kg) * (vrE / v_rel)
            aD_N = -(Fd / proj.mass_kg) * (vrN / v_rel)
            aD_U = -(Fd / proj.mass_kg) * (vrU / v_rel)
        else:
            aD_E = aD_N = aD_U = 0.0

        # Ефект Коріоліса (для великих дальностей)
        if self._coriolis:
            lat_rad = math.radians(lat_deg)
            Ω = OMEGA_EARTH
            # Спрощена формула (тільки горизонтальні компоненти)
            a_cor_E =  2.0 * Ω * (vN * math.sin(lat_rad) - vU * math.cos(lat_rad))
            a_cor_N = -2.0 * Ω * vE * math.sin(lat_rad)
            a_cor_U =  2.0 * Ω * vE * math.cos(lat_rad)
        else:
            a_cor_E = a_cor_N = a_cor_U = 0.0

        # Прискорення
        aE = aD_E + a_cor_E
        aN = aD_N + a_cor_N
        aU = aD_U + a_cor_U - G   # гравітація

        return (vE, vN, vU, aE, aN, aU)

    def _adaptive_dt(self, height_above_origin: float) -> float:
        """
        Адаптивний часовий крок.
        Менший крок поблизу рельєфу для точнішого hit detection.
        """
        if height_above_origin < 50.0:
            return self._dt * 0.1
        if height_above_origin < 200.0:
            return self._dt * 0.3
        return self._dt

    def _terrain_height_at(
        self,
        east_m:     float,
        north_m:    float,
        origin:     LatLonAlt,
        origin_asl: float,
    ) -> float | None:
        """
        Отримати висоту рельєфу в точці ENU.
        Повертає висоту ASL або None якщо поза DEM.
        """
        if self._dem is None:
            return 0.0   # Без DEM — плоский рельєф

        R      = R_EARTH
        cosLat = math.cos(math.radians(origin.lat))

        lat = origin.lat + math.degrees(north_m / R)
        lon = origin.lon + math.degrees(east_m / (R * cosLat))

        h = self._dem.sample(lat, lon)
        return float(h) if h is not None else None

    def _interpolate_impact(
        self,
        e0: float, n0: float, u0: float,
        e1: float, n1: float, u1: float,
        origin: LatLonAlt,
        origin_asl: float,
    ) -> tuple[LatLonAlt, float]:
        """
        Лінійна інтерполяція точки зіткнення з рельєфом.
        Бінарний пошук між двома точками.
        """
        lo, hi = 0.0, 1.0
        for _ in range(16):   # 16 ітерацій → точність ~1мм
            mid = (lo + hi) * 0.5
            em  = e0 + (e1 - e0) * mid
            nm  = n0 + (n1 - n0) * mid
            um  = u0 + (u1 - u0) * mid

            terrain = self._terrain_height_at(em, nm, origin, origin_asl)
            if terrain is None:
                break

            current_asl = origin_asl + um
            if current_asl >= terrain:
                lo = mid
            else:
                hi = mid

        # Фінальна точка
        t  = (lo + hi) * 0.5
        ef = e0 + (e1 - e0) * t
        nf = n0 + (n1 - n0) * t
        uf = u0 + (u1 - u0) * t

        terrain_h = self._terrain_height_at(ef, nf, origin, origin_asl) or origin_asl
        impact = _enu_to_latlon_single(
            origin.lat, origin.lon, origin.alt,
            ef, nf, uf,
        )
        return impact, uf


# ── ДОПОМІЖНІ ФУНКЦІЇ ──────────────────────────────────────────

def _enu_to_latlon_single(
    origin_lat: float, origin_lon: float, origin_alt: float,
    east: float, north: float, up: float,
) -> LatLonAlt:
    R      = R_EARTH
    cosLat = math.cos(math.radians(origin_lat))
    lat    = origin_lat + math.degrees(north / R)
    lon    = origin_lon + math.degrees(east  / (R * cosLat))
    alt    = origin_alt + up
    return LatLonAlt(lat=lat, lon=lon, alt=alt)


def _enu_to_latlon(
    origin_lat: float, origin_lon: float, origin_alt: float,
    east_arr:  npt.NDArray,
    north_arr: npt.NDArray,
    up_arr:    npt.NDArray,
) -> list[LatLonAlt]:
    R      = R_EARTH
    cosLat = math.cos(math.radians(origin_lat))
    result = []
    for e, n, u in zip(east_arr, north_arr, up_arr, strict=True):
        result.append(LatLonAlt(
            lat = origin_lat + math.degrees(n / R),
            lon = origin_lon + math.degrees(e / (R * cosLat)),
            alt = origin_alt + u,
        ))
    return result


def optimal_elevation(
    range_m:         float,
    muzzle_velocity: float,
    high_angle:      bool = False,
) -> float:
    """
    Оптимальний кут підвищення для заданої дальності
    (без опору повітря — аналітична формула).

    Args:
        range_m:         цільова дальність, м
        muzzle_velocity: початкова швидкість, м/с
        high_angle:      True = верхня траєкторія (> 45°)

    Returns:
        Кут підвищення у градусах або None якщо недосяжно
    """
    # R = v0² × sin(2α) / g  →  sin(2α) = R×g / v0²
    sin2a = (range_m * G) / (muzzle_velocity ** 2)
    if abs(sin2a) > 1.0:
        return None   # недосяжна дальність

    a = math.degrees(math.asin(sin2a)) / 2.0   # 0..45°
    return 90.0 - a if high_angle else a
