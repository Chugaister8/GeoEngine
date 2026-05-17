"""
GeoEngine — Camera State
Python-side представлення камери рендерера.

Синхронізується з JS GeoRenderer через WebSocket.
Використовується для:
  - Серверного frustum culling
  - LOD вибору на сервері
  - Запису/відтворення анімацій
  - Bookmarks / waypoints
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..geo.coords import LLH, haversine_distance
from ..utils.math3d import Vec3, Quat, Mat4, deg_to_rad


# ----------------------------------------------------------------
# CAMERA STATE
# ----------------------------------------------------------------

@dataclass(slots=True)
class CameraState:
    """
    Стан камери у географічних координатах.

    Поля:
        lat, lon, alt:  позиція (WGS84)
        heading:        азимут (0=Північ, 90=Схід, градуси)
        pitch:          нахил (-90=вниз, 0=горизонт, 90=вгору)
        fov:            field of view (градуси)
        near, far:      clip planes (метри)
    """
    lat:     float = 48.25
    lon:     float = 23.50
    alt:     float = 5000.0
    heading: float = 0.0
    pitch:   float = -30.0
    fov:     float = 60.0
    near:    float = 1.0
    far:     float = 10_000_000.0

    def __post_init__(self) -> None:
        self.heading = self.heading % 360.0
        self.pitch   = max(-90.0, min(90.0, self.pitch))
        self.fov     = max(10.0,  min(170.0, self.fov))
        self.alt     = max(0.1, self.alt)

    @property
    def llh(self) -> LLH:
        return LLH(lat=self.lat, lon=self.lon, alt=self.alt)

    @property
    def aspect_ratio(self) -> float:
        """Співвідношення сторін (встановлюється при рендерингу)."""
        return 16.0 / 9.0   # дефолт HD

    def distance_to(self, other: "CameraState") -> float:
        """Відстань до іншої позиції камери (метри)."""
        return haversine_distance(self.llh, other.llh)

    def to_dict(self) -> dict:
        return {
            "lat":     self.lat,
            "lon":     self.lon,
            "alt":     self.alt,
            "heading": self.heading,
            "pitch":   self.pitch,
            "fov":     self.fov,
            "near":    self.near,
            "far":     self.far,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraState":
        return cls(
            lat=float(d.get("lat", 48.25)),
            lon=float(d.get("lon", 23.50)),
            alt=float(d.get("alt", 5000.0)),
            heading=float(d.get("heading", 0.0)),
            pitch=float(d.get("pitch", -30.0)),
            fov=float(d.get("fov", 60.0)),
            near=float(d.get("near", 1.0)),
            far=float(d.get("far", 10_000_000.0)),
        )

    def __repr__(self) -> str:
        return (
            f"Camera(lat={self.lat:.4f}, lon={self.lon:.4f}, "
            f"alt={self.alt:.0f}m, heading={self.heading:.1f}°)"
        )


# ----------------------------------------------------------------
# CAMERA BOOKMARK
# ----------------------------------------------------------------

@dataclass
class CameraBookmark:
    """Збережена позиція камери (waypoint)."""
    name:        str
    state:       CameraState
    description: str = ""
    tags:        list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "description": self.description,
            "tags":        self.tags,
            "camera":      self.state.to_dict(),
        }


# ----------------------------------------------------------------
# CAMERA ANIMATION
# ----------------------------------------------------------------

@dataclass
class CameraKeyframe:
    """Один кадр анімації камери."""
    time:  float          # секунди
    state: CameraState
    easing: str = "ease_in_out"  # linear / ease_in / ease_out / ease_in_out


class CameraAnimation:
    """
    Анімація камери вздовж шляху (кілька keyframes).

    Usage:
        anim = CameraAnimation()
        anim.add_keyframe(0.0, CameraState(lat=48.0, lon=23.0, alt=5000))
        anim.add_keyframe(3.0, CameraState(lat=48.5, lon=24.0, alt=2000))
        state = anim.evaluate(1.5)  # → інтерпольований стан
    """

    def __init__(self) -> None:
        self._keyframes: list[CameraKeyframe] = []

    def add_keyframe(
        self,
        time:   float,
        state:  CameraState,
        easing: str = "ease_in_out",
    ) -> "CameraAnimation":
        """Додати ключовий кадр."""
        kf = CameraKeyframe(time=time, state=state, easing=easing)
        self._keyframes.append(kf)
        self._keyframes.sort(key=lambda k: k.time)
        return self

    @property
    def duration(self) -> float:
        """Загальна тривалість (секунди)."""
        if not self._keyframes:
            return 0.0
        return self._keyframes[-1].time

    def evaluate(self, time: float) -> CameraState:
        """
        Обчислити стан камери у момент часу t.

        Використовує лінійну інтерполяцію між keyframes.
        """
        if not self._keyframes:
            return CameraState()

        time = max(0.0, min(self.duration, time))

        # Знайти два сусідніх keyframe
        for i, kf in enumerate(self._keyframes):
            if kf.time >= time:
                if i == 0:
                    return kf.state
                prev_kf = self._keyframes[i - 1]
                # Нормалізований t між двома кадрами
                dt  = kf.time - prev_kf.time
                t   = (time - prev_kf.time) / dt if dt > 0 else 0.0
                t   = _apply_easing(t, kf.easing)
                return _interpolate_camera(prev_kf.state, kf.state, t)

        return self._keyframes[-1].state

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "keyframes": [
                {"time": kf.time, "easing": kf.easing, **kf.state.to_dict()}
                for kf in self._keyframes
            ],
        }


def _interpolate_camera(a: CameraState, b: CameraState, t: float) -> CameraState:
    """Лінійна інтерполяція між двома станами камери."""
    def lerp(x: float, y: float) -> float:
        return x + (y - x) * t

    # Heading — кутова інтерполяція (через коротший шлях)
    dh = ((b.heading - a.heading + 180) % 360) - 180
    heading = (a.heading + dh * t) % 360

    return CameraState(
        lat=lerp(a.lat, b.lat),
        lon=lerp(a.lon, b.lon),
        alt=lerp(a.alt, b.alt),
        heading=heading,
        pitch=lerp(a.pitch, b.pitch),
        fov=lerp(a.fov, b.fov),
        near=lerp(a.near, b.near),
        far=lerp(a.far, b.far),
    )


def _apply_easing(t: float, easing: str) -> float:
    """Застосувати easing функцію до t ∈ [0, 1]."""
    match easing:
        case "linear":
            return t
        case "ease_in":
            return t * t
        case "ease_out":
            return 1 - (1 - t) ** 2
        case "ease_in_out":
            return t * t * (3 - 2 * t)   # smoothstep
        case _:
            return t
