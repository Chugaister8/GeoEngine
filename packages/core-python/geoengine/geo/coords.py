"""
GeoEngine — Coordinate Transformations
Перетворення між системами координат без зовнішніх залежностей
для простих операцій; pyproj для складних.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# ----------------------------------------------------------------
# WGS84 константи
# ----------------------------------------------------------------

_A: Final[float] = 6_378_137.0          # велика піввісь (м)
_B: Final[float] = 6_356_752.314245     # мала піввісь  (м)
_F: Final[float] = 1.0 / 298.257223563  # стиснення
_E2: Final[float] = 1.0 - (_B / _A) ** 2  # ексцентриситет²
_E: Final[float] = math.sqrt(_E2)          # ексцентриситет

DEG2RAD: Final[float] = math.pi / 180.0
RAD2DEG: Final[float] = 180.0 / math.pi


# ----------------------------------------------------------------
# ТИПИ ТОЧОК
# ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LLH:
    """
    Географічна точка WGS84:
    lat (°), lon (°), height над еліпсоїдом (м).
    """
    lat: float   # -90..90
    lon: float   # -180..180
    alt: float = 0.0  # метри над еліпсоїдом

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"lat={self.lat} поза [-90, 90]")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"lon={self.lon} поза [-180, 180]")


@dataclass(frozen=True, slots=True)
class ECEF:
    """
    Earth-Centered Earth-Fixed (метри).
    Початок — центр Землі.
    X: через (0°lat, 0°lon)
    Y: через (0°lat, 90°lon)
    Z: через Північний полюс
    """
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class ENU:
    """
    East-North-Up (метри) відносно опорної точки.
    Використовується як локальна система координат рушія.
    """
    east:  float
    north: float
    up:    float


@dataclass(frozen=True, slots=True)
class WebMercator:
    """
    EPSG:3857 — WebMercator проекція (метри).
    Використовується тайловими картами.
    """
    x: float   # -20026376..20026376
    y: float   # -20048966..20048966


# ----------------------------------------------------------------
# LLH ↔ ECEF
# ----------------------------------------------------------------

def llh_to_ecef(point: LLH) -> ECEF:
    """
    Географічна → ECEF (Earth-Centered, Earth-Fixed).

    Алгоритм: стандартна формула через радіус кривини N(φ).
    Точність: субміліметрова.
    """
    lat_r = math.radians(point.lat)
    lon_r = math.radians(point.lon)
    h     = point.alt

    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    sin_lon = math.sin(lon_r)
    cos_lon = math.cos(lon_r)

    # Радіус кривини у першому вертикалі
    N = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - _E2) + h) * sin_lat

    return ECEF(x=x, y=y, z=z)


def ecef_to_llh(point: ECEF) -> LLH:
    """
    ECEF → Географічна WGS84.

    Алгоритм: Bowring ітераційний (збіжність < 3 ітерацій для ≤9000км висоти).
    """
    x, y, z = point.x, point.y, point.z
    p = math.sqrt(x * x + y * y)   # відстань від осі Z

    # Початкове наближення
    lon = math.atan2(y, x)

    # Ітераційний метод Бовринга
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
        lat_new = math.atan2(z + _E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:  # ~0.001 мм
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    N = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) / abs(sin_lat) - N * (1.0 - _E2)

    return LLH(
        lat=math.degrees(lat),
        lon=math.degrees(lon),
        alt=h,
    )


# ----------------------------------------------------------------
# LLH ↔ ENU (локальна система рушія)
# ----------------------------------------------------------------

def llh_to_enu(point: LLH, origin: LLH) -> ENU:
    """
    Перетворити LLH точку в локальну ENU відносно origin.

    ENU (East-North-Up) використовується як world-space рушія.
    Для невеликих регіонів (~100км) похибка < 1мм.

    Args:
        point:  точка яку конвертуємо
        origin: початок локальної системи координат

    Returns:
        ENU координати в метрах від origin
    """
    ecef_p = llh_to_ecef(point)
    ecef_o = llh_to_ecef(origin)

    # Різниця в ECEF
    dx = ecef_p.x - ecef_o.x
    dy = ecef_p.y - ecef_o.y
    dz = ecef_p.z - ecef_o.z

    # Матриця обертання ECEF → ENU
    lat_r = math.radians(origin.lat)
    lon_r = math.radians(origin.lon)

    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    sin_lon = math.sin(lon_r)
    cos_lon = math.cos(lon_r)

    # Рядки матриці обертання:
    # East:  [-sin_lon,          cos_lon,         0       ]
    # North: [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat ]
    # Up:    [ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat ]

    east  = -sin_lon * dx + cos_lon * dy
    north = (-sin_lat * cos_lon * dx
             - sin_lat * sin_lon * dy
             + cos_lat * dz)
    up    = (cos_lat * cos_lon * dx
             + cos_lat * sin_lon * dy
             + sin_lat * dz)

    return ENU(east=east, north=north, up=up)


def enu_to_llh(enu: ENU, origin: LLH) -> LLH:
    """
    ENU (локальна) → LLH WGS84.

    Args:
        enu:    точка в локальній ENU системі
        origin: початок ENU системи

    Returns:
        LLH точка у WGS84
    """
    lat_r = math.radians(origin.lat)
    lon_r = math.radians(origin.lon)

    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    sin_lon = math.sin(lon_r)
    cos_lon = math.cos(lon_r)

    ecef_o = llh_to_ecef(origin)

    # Зворотня матриця (транспонована, бо ортогональна)
    # [East, North, Up] → [X, Y, Z] delta
    dx = (-sin_lon    * enu.east
          - sin_lat * cos_lon * enu.north
          + cos_lat * cos_lon * enu.up)
    dy = ( cos_lon    * enu.east
          - sin_lat * sin_lon * enu.north
          + cos_lat * sin_lon * enu.up)
    dz = ( cos_lat * enu.north
          + sin_lat * enu.up)

    ecef_p = ECEF(
        x=ecef_o.x + dx,
        y=ecef_o.y + dy,
        z=ecef_o.z + dz,
    )
    return ecef_to_llh(ecef_p)


# ----------------------------------------------------------------
# LLH ↔ WebMercator
# ----------------------------------------------------------------

def llh_to_webmercator(point: LLH) -> WebMercator:
    """
    LLH WGS84 → WebMercator (EPSG:3857, метри).

    ⚠️  WebMercator не зберігає площі та форми при великих широтах.
    Максимальна latitude: ≈85.051129°.
    """
    lat_clamp = max(-85.051129, min(85.051129, point.lat))
    x = _A * math.radians(point.lon)
    y = _A * math.log(
        math.tan(math.pi / 4.0 + math.radians(lat_clamp) / 2.0)
    )
    return WebMercator(x=x, y=y)


def webmercator_to_llh(point: WebMercator, alt: float = 0.0) -> LLH:
    """WebMercator (EPSG:3857) → LLH WGS84."""
    lon = math.degrees(point.x / _A)
    lat = math.degrees(
        2.0 * math.atan(math.exp(point.y / _A)) - math.pi / 2.0
    )
    return LLH(lat=lat, lon=lon, alt=alt)


# ----------------------------------------------------------------
# ВІДСТАНІ
# ----------------------------------------------------------------

def haversine_distance(a: LLH, b: LLH) -> float:
    """
    Відстань між двома точками по поверхні сфери (метри).
    Формула Гаверсинуса.

    Точність: ~0.5% (сфера, не еліпсоїд).
    Для геодезичної точності — використовуй vincenty_distance.

    Args:
        a, b: дві точки LLH (висота ігнорується)

    Returns:
        Відстань у метрах
    """
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)

    sin_dlat = math.sin(dlat / 2)
    sin_dlon = math.sin(dlon / 2)

    h = (sin_dlat * sin_dlat
         + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon)

    return 2.0 * _A * math.asin(math.sqrt(h))


def vincenty_distance(a: LLH, b: LLH) -> float:
    """
    Геодезична відстань між двома точками на еліпсоїді WGS84.
    Формула Вінченті — точність ~0.06 мм.

    Args:
        a, b: дві точки LLH (висота ігнорується)

    Returns:
        Відстань у метрах

    Raises:
        ValueError: якщо точки антиподальні (збіжність не досягнута)
    """
    if abs(a.lat - b.lat) < 1e-10 and abs(a.lon - b.lon) < 1e-10:
        return 0.0

    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    L    = math.radians(b.lon - a.lon)

    U1 = math.atan((1 - _F) * math.tan(lat1))
    U2 = math.atan((1 - _F) * math.tan(lat2))

    sin_U1, cos_U1 = math.sin(U1), math.cos(U1)
    sin_U2, cos_U2 = math.sin(U2), math.cos(U2)

    lam = L
    for _ in range(1000):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)

        sin_sigma = math.sqrt(
            (cos_U2 * sin_lam) ** 2
            + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam) ** 2
        )
        if sin_sigma == 0:
            return 0.0  # збігаються

        cos_sigma  = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
        sigma      = math.atan2(sin_sigma, cos_sigma)
        sin_alpha  = cos_U1 * cos_U2 * sin_lam / sin_sigma
        cos2_alpha = 1.0 - sin_alpha ** 2
        cos_2sm    = (cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha
                      if cos2_alpha != 0 else 0.0)
        C = _F / 16.0 * cos2_alpha * (4.0 + _F * (4.0 - 3.0 * cos2_alpha))

        lam_new = L + (1.0 - C) * _F * sin_alpha * (
            sigma + C * sin_sigma * (
                cos_2sm + C * cos_sigma * (-1.0 + 2.0 * cos_2sm ** 2)
            )
        )
        if abs(lam_new - lam) < 1e-12:
            break
        lam = lam_new
    else:
        raise ValueError("Vincenty не збігся — можливо антиподальні точки")

    u2 = cos2_alpha * (_A ** 2 - _B ** 2) / _B ** 2
    A_coef = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B_coef = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))

    delta_sigma = B_coef * sin_sigma * (
        cos_2sm + B_coef / 4.0 * (
            cos_sigma * (-1.0 + 2.0 * cos_2sm ** 2)
            - B_coef / 6.0 * cos_2sm * (-3.0 + 4.0 * sin_sigma ** 2)
            * (-3.0 + 4.0 * cos_2sm ** 2)
        )
    )
    return _B * A_coef * (sigma - delta_sigma)


def bearing(a: LLH, b: LLH) -> float:
    """
    Початковий азимут від точки a до b (градуси від Північ, за годинниковою).

    Returns:
        Кут у градусах [0, 360)
    """
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)

    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

    bearing_deg = math.degrees(math.atan2(x, y))
    return (bearing_deg + 360.0) % 360.0
