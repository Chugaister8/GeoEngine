"""
GeoEngine — Coordinate Transformation Tests
"""

from __future__ import annotations

import math
import pytest

from geoengine.geo.coords import (
    LLH, ECEF, ENU, WebMercator,
    llh_to_ecef, ecef_to_llh,
    llh_to_enu, enu_to_llh,
    llh_to_webmercator, webmercator_to_llh,
    haversine_distance, vincenty_distance, bearing,
    DEG2RAD, RAD2DEG,
)


# ================================================================
# LLH VALIDATION
# ================================================================

class TestLLH:

    def test_valid_point(self):
        p = LLH(lat=48.0, lon=23.0, alt=200.0)
        assert p.lat == 48.0
        assert p.lon == 23.0
        assert p.alt == 200.0

    def test_default_alt_zero(self):
        p = LLH(lat=0.0, lon=0.0)
        assert p.alt == 0.0

    def test_invalid_lat_too_high(self):
        with pytest.raises(ValueError, match="lat"):
            LLH(lat=91.0, lon=0.0)

    def test_invalid_lat_too_low(self):
        with pytest.raises(ValueError, match="lat"):
            LLH(lat=-91.0, lon=0.0)

    def test_invalid_lon(self):
        with pytest.raises(ValueError, match="lon"):
            LLH(lat=0.0, lon=181.0)

    def test_boundary_values(self):
        # Граничні значення не кидають виключень
        LLH(lat=90.0, lon=180.0)
        LLH(lat=-90.0, lon=-180.0)
        LLH(lat=0.0, lon=0.0)


# ================================================================
# LLH ↔ ECEF
# ================================================================

class TestLLHtoECEF:

    def test_equator_prime_meridian(self):
        """(0°, 0°, 0m) → (6378137, 0, 0) на екваторі."""
        ecef = llh_to_ecef(LLH(lat=0.0, lon=0.0, alt=0.0))
        assert ecef.x == pytest.approx(6_378_137.0, rel=1e-6)
        assert ecef.y == pytest.approx(0.0, abs=1.0)
        assert ecef.z == pytest.approx(0.0, abs=1.0)

    def test_north_pole(self):
        """Північний полюс → Z = b (мала піввісь)."""
        ecef = llh_to_ecef(LLH(lat=90.0, lon=0.0, alt=0.0))
        assert ecef.x == pytest.approx(0.0, abs=1.0)
        assert ecef.y == pytest.approx(0.0, abs=1.0)
        assert ecef.z == pytest.approx(6_356_752.3, rel=1e-5)

    def test_equator_90_lon(self):
        """(0°, 90°, 0m) → Y = R."""
        ecef = llh_to_ecef(LLH(lat=0.0, lon=90.0, alt=0.0))
        assert ecef.x == pytest.approx(0.0, abs=1.0)
        assert ecef.y == pytest.approx(6_378_137.0, rel=1e-6)

    def test_round_trip_kyiv(self):
        """LLH → ECEF → LLH зберігає точність."""
        original = LLH(lat=50.45, lon=30.52, alt=200.0)
        ecef     = llh_to_ecef(original)
        restored = ecef_to_llh(ecef)

        assert restored.lat == pytest.approx(original.lat, abs=1e-8)
        assert restored.lon == pytest.approx(original.lon, abs=1e-8)
        assert restored.alt == pytest.approx(original.alt, abs=1e-3)

    def test_round_trip_hoverla(self):
        """Говерла (найвища точка України)."""
        original = LLH(lat=48.16, lon=24.50, alt=2061.0)
        ecef     = llh_to_ecef(original)
        restored = ecef_to_llh(ecef)

        assert restored.lat == pytest.approx(original.lat, abs=1e-8)
        assert restored.lon == pytest.approx(original.lon, abs=1e-8)
        assert restored.alt == pytest.approx(original.alt, abs=0.01)

    def test_round_trip_south_pole(self):
        original = LLH(lat=-90.0, lon=0.0, alt=2835.0)
        ecef     = llh_to_ecef(original)
        restored = ecef_to_llh(ecef)
        assert restored.lat == pytest.approx(-90.0, abs=1e-6)
        assert restored.alt == pytest.approx(2835.0, abs=1.0)


# ================================================================
# LLH ↔ ENU
# ================================================================

class TestLLHtoENU:

    def test_origin_is_zero(self):
        """Точка в origin → ENU = (0,0,0)."""
        origin = LLH(lat=48.0, lon=23.0, alt=0.0)
        enu    = llh_to_enu(origin, origin)
        assert enu.east  == pytest.approx(0.0, abs=1e-3)
        assert enu.north == pytest.approx(0.0, abs=1e-3)
        assert enu.up    == pytest.approx(0.0, abs=1e-3)

    def test_east_direction(self):
        """Точка на схід → positive East."""
        origin = LLH(lat=48.0, lon=23.0, alt=0.0)
        east   = LLH(lat=48.0, lon=23.01, alt=0.0)  # ~800м на схід
        enu    = llh_to_enu(east, origin)

        assert enu.east  > 0
        assert abs(enu.north) < 1.0    # нульовий north
        assert abs(enu.up)    < 0.01   # нульовий up

    def test_north_direction(self):
        """Точка на північ → positive North."""
        origin = LLH(lat=48.0, lon=23.0, alt=0.0)
        north  = LLH(lat=48.01, lon=23.0, alt=0.0)  # ~1112м на північ
        enu    = llh_to_enu(north, origin)

        assert enu.north > 0
        assert abs(enu.east) < 1.0

    def test_up_direction(self):
        """Точка вище → positive Up."""
        origin = LLH(lat=48.0, lon=23.0, alt=0.0)
        above  = LLH(lat=48.0, lon=23.0, alt=1000.0)
        enu    = llh_to_enu(above, origin)

        assert enu.up    == pytest.approx(1000.0, rel=1e-4)
        assert abs(enu.east)  < 1.0
        assert abs(enu.north) < 1.0

    def test_round_trip_enu(self):
        """LLH → ENU → LLH зберігає точність (~1мм)."""
        origin = LLH(lat=48.0, lon=23.0, alt=0.0)
        point  = LLH(lat=48.05, lon=23.1, alt=500.0)

        enu      = llh_to_enu(point, origin)
        restored = enu_to_llh(enu, origin)

        assert restored.lat == pytest.approx(point.lat, abs=1e-6)
        assert restored.lon == pytest.approx(point.lon, abs=1e-6)
        assert restored.alt == pytest.approx(point.alt, abs=0.01)

    def test_enu_distance_matches_haversine(self):
        """Відстань ENU ≈ haversine для малих відстаней."""
        origin = LLH(lat=48.0, lon=23.0, alt=0.0)
        point  = LLH(lat=48.05, lon=23.05, alt=0.0)

        enu  = llh_to_enu(point, origin)
        enu_dist = math.sqrt(enu.east**2 + enu.north**2)
        hav_dist = haversine_distance(origin, point)

        assert enu_dist == pytest.approx(hav_dist, rel=0.001)  # < 0.1%


# ================================================================
# WebMercator
# ================================================================

class TestWebMercator:

    def test_equator_prime_meridian(self):
        """(0°, 0°) → (0, 0) в WebMercator."""
        wm = llh_to_webmercator(LLH(lat=0.0, lon=0.0))
        assert wm.x == pytest.approx(0.0, abs=1.0)
        assert wm.y == pytest.approx(0.0, abs=1.0)

    def test_kyiv(self):
        """Київ має positive X та positive Y."""
        kyiv = LLH(lat=50.45, lon=30.52)
        wm   = llh_to_webmercator(kyiv)
        assert wm.x > 0
        assert wm.y > 0

    def test_round_trip(self):
        """LLH → WebMercator → LLH зберігає точність."""
        original = LLH(lat=48.0, lon=23.0)
        wm       = llh_to_webmercator(original)
        restored = webmercator_to_llh(wm)

        assert restored.lat == pytest.approx(original.lat, abs=1e-6)
        assert restored.lon == pytest.approx(original.lon, abs=1e-6)

    def test_max_latitude_clamped(self):
        """Широта > 85.051129° обрізається."""
        extreme = LLH(lat=89.0, lon=0.0)
        wm      = llh_to_webmercator(extreme)
        # Не має бути нескінченності
        assert math.isfinite(wm.x)
        assert math.isfinite(wm.y)


# ================================================================
# ВІДСТАНІ
# ================================================================

class TestDistances:

    # Haversine

    def test_haversine_same_point(self):
        p = LLH(lat=48.0, lon=23.0)
        assert haversine_distance(p, p) == pytest.approx(0.0, abs=1e-3)

    def test_haversine_kyiv_lviv(self):
        """Відстань Київ–Львів ≈ 470км."""
        kyiv = LLH(lat=50.45, lon=30.52)
        lviv = LLH(lat=49.84, lon=24.02)
        dist = haversine_distance(kyiv, lviv)
        assert dist == pytest.approx(470_000, rel=0.05)  # ±5%

    def test_haversine_equator_1_degree(self):
        """1° на екваторі ≈ 111320м."""
        a = LLH(lat=0.0, lon=0.0)
        b = LLH(lat=0.0, lon=1.0)
        assert haversine_distance(a, b) == pytest.approx(111_320, rel=0.01)

    # Vincenty

    def test_vincenty_same_point(self):
        p = LLH(lat=48.0, lon=23.0)
        assert vincenty_distance(p, p) == pytest.approx(0.0, abs=1e-3)

    def test_vincenty_vs_haversine_small_distance(self):
        """Для малих відстаней vincenty ≈ haversine."""
        a = LLH(lat=48.0, lon=23.0)
        b = LLH(lat=48.01, lon=23.01)

        hav = haversine_distance(a, b)
        vin = vincenty_distance(a, b)

        assert abs(hav - vin) / vin < 0.001   # < 0.1% різниця

    def test_vincenty_more_accurate_long_distance(self):
        """Vincenty точніший для великих відстаней."""
        # Відстань між Лондоном та Сіднеєм
        london = LLH(lat=51.5, lon=-0.1)
        sydney = LLH(lat=-33.9, lon=151.2)

        dist = vincenty_distance(london, sydney)
        # ~17000 км
        assert dist == pytest.approx(17_000_000, rel=0.01)

    # Bearing

    def test_bearing_east(self):
        """Точка на схід → bearing ≈ 90°."""
        a = LLH(lat=48.0, lon=23.0)
        b = LLH(lat=48.0, lon=24.0)
        b_deg = bearing(a, b)
        assert b_deg == pytest.approx(90.0, abs=1.0)

    def test_bearing_north(self):
        """Точка на північ → bearing ≈ 0°."""
        a = LLH(lat=48.0, lon=23.0)
        b = LLH(lat=49.0, lon=23.0)
        b_deg = bearing(a, b)
        assert b_deg == pytest.approx(0.0, abs=1.0)

    def test_bearing_south(self):
        """Точка на південь → bearing ≈ 180°."""
        a = LLH(lat=48.0, lon=23.0)
        b = LLH(lat=47.0, lon=23.0)
        b_deg = bearing(a, b)
        assert b_deg == pytest.approx(180.0, abs=1.0)

    def test_bearing_range(self):
        """Bearing завжди в [0, 360)."""
        points = [
            (LLH(lat=0, lon=0), LLH(lat=1, lon=1)),
            (LLH(lat=0, lon=0), LLH(lat=-1, lon=-1)),
            (LLH(lat=0, lon=0), LLH(lat=1, lon=-1)),
        ]
        for a, b in points:
            b_deg = bearing(a, b)
            assert 0.0 <= b_deg < 360.0
