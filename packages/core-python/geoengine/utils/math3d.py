"""
GeoEngine — Math3D
3D математичні примітиви для Python core.

Використовується в:
  scene/node.py    — Transform, SceneNode.world_transform()
  mesh/terrain.py  — нормалі, AABB
  dem/analysis.py  — векторні обчислення

Всі типи immutable (frozen dataclass або tuple-based)
для безпечного використання у multi-threaded контексті.

Координатна система: Three.js конвенція
  X = East
  Y = Up
  Z = -North
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence


# ────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────

PI      = math.pi
TWO_PI  = 2.0 * math.pi
HALF_PI = math.pi * 0.5
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
EPSILON = 1e-10


def deg_to_rad(deg: float) -> float:
    return deg * DEG2RAD

def rad_to_deg(rad: float) -> float:
    return rad * RAD2DEG

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0 + EPSILON), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def sign(x: float) -> float:
    if x > 0: return  1.0
    if x < 0: return -1.0
    return 0.0


# ────────────────────────────────────────────────────────────────
# VEC2
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Vec2:
    """2D вектор."""
    x: float = 0.0
    y: float = 0.0

    # ---- Арифметика ----
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    # ---- Властивості ----
    @property
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    # ---- Методи ----
    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def normalized(self) -> "Vec2":
        l = self.length
        if l < EPSILON:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / l, self.y / l)

    def lerp(self, other: "Vec2", t: float) -> "Vec2":
        return Vec2(
            lerp(self.x, other.x, t),
            lerp(self.y, other.y, t),
        )

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    # ---- Конструктори ----
    @classmethod
    def zero(cls) -> "Vec2":
        return cls(0.0, 0.0)

    @classmethod
    def one(cls) -> "Vec2":
        return cls(1.0, 1.0)

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> "Vec2":
        return cls(t[0], t[1])

    def __repr__(self) -> str:
        return f"Vec2({self.x:.4f}, {self.y:.4f})"


# ────────────────────────────────────────────────────────────────
# VEC3
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Vec3:
    """3D вектор."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # ---- Арифметика ----
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    # ---- Властивості ----
    @property
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def length_sq(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    # ---- Методи ----
    def dot(self, other: "Vec3") -> float:
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def normalized(self) -> "Vec3":
        l = self.length
        if l < EPSILON:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        return Vec3(
            lerp(self.x, other.x, t),
            lerp(self.y, other.y, t),
            lerp(self.z, other.z, t),
        )

    def reflect(self, normal: "Vec3") -> "Vec3":
        """Відбиття вектора від нормалі."""
        d = 2.0 * self.dot(normal)
        return Vec3(
            self.x - d * normal.x,
            self.y - d * normal.y,
            self.z - d * normal.z,
        )

    def project(self, onto: "Vec3") -> "Vec3":
        """Проєкція на інший вектор."""
        d = onto.dot(onto)
        if d < EPSILON:
            return Vec3.zero()
        s = self.dot(onto) / d
        return onto * s

    def angle_to(self, other: "Vec3") -> float:
        """Кут між векторами у радіанах."""
        d = clamp(self.normalized().dot(other.normalized()), -1.0, 1.0)
        return math.acos(d)

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).length

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    # ---- Конструктори ----
    @classmethod
    def zero(cls) -> "Vec3":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def one(cls) -> "Vec3":
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def up(cls) -> "Vec3":
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def forward(cls) -> "Vec3":
        return cls(0.0, 0.0, -1.0)

    @classmethod
    def right(cls) -> "Vec3":
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def from_tuple(cls, t: Sequence[float]) -> "Vec3":
        return cls(float(t[0]), float(t[1]), float(t[2]))

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


# ────────────────────────────────────────────────────────────────
# VEC4
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Vec4:
    """4D вектор / homogeneous coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __add__(self, other: "Vec4") -> "Vec4":
        return Vec4(self.x+other.x, self.y+other.y,
                    self.z+other.z, self.w+other.w)

    def __mul__(self, s: float) -> "Vec4":
        return Vec4(self.x*s, self.y*s, self.z*s, self.w*s)

    @property
    def xyz(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def zero(cls) -> "Vec4":
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_vec3(cls, v: Vec3, w: float = 1.0) -> "Vec4":
        return cls(v.x, v.y, v.z, w)

    def __repr__(self) -> str:
        return f"Vec4({self.x:.4f}, {self.y:.4f}, {self.z:.4f}, {self.w:.4f})"


# ────────────────────────────────────────────────────────────────
# QUATERNION
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Quat:
    """
    Кватерніон для представлення обертань.

    Нормалізований кватерніон: x²+y²+z²+w² = 1
    Convention: w — скалярна частина (Hamilton)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    # ---- Арифметика ----
    def __mul__(self, other: "Quat") -> "Quat":
        """Множення кватерніонів (composition of rotations)."""
        return Quat(
            x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
            w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
        )

    def __neg__(self) -> "Quat":
        return Quat(-self.x, -self.y, -self.z, -self.w)

    # ---- Властивості ----
    @property
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    @property
    def conjugate(self) -> "Quat":
        return Quat(-self.x, -self.y, -self.z, self.w)

    @property
    def inverse(self) -> "Quat":
        l2 = self.x**2 + self.y**2 + self.z**2 + self.w**2
        if l2 < EPSILON:
            return Quat.identity()
        return Quat(-self.x/l2, -self.y/l2, -self.z/l2, self.w/l2)

    # ---- Методи ----
    def normalized(self) -> "Quat":
        l = self.length
        if l < EPSILON:
            return Quat.identity()
        return Quat(self.x/l, self.y/l, self.z/l, self.w/l)

    def rotate_vec3(self, v: Vec3) -> Vec3:
        """Обернути вектор цим кватерніоном."""
        # Оптимізована формула: q * v * q^-1
        qv = Vec3(self.x, self.y, self.z)
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        uv  = uv  * (2.0 * self.w)
        uuv = uuv * 2.0
        return Vec3(
            v.x + uv.x + uuv.x,
            v.y + uv.y + uuv.y,
            v.z + uv.z + uuv.z,
        )

    def slerp(self, other: "Quat", t: float) -> "Quat":
        """Spherical linear interpolation."""
        dot = (self.x*other.x + self.y*other.y +
               self.z*other.z + self.w*other.w)

        # Найкоротший шлях
        other_ = other if dot >= 0.0 else -other
        dot     = abs(dot)

        if dot > 0.9995:
            # Лінійна інтерполяція для близьких кватерніонів
            result = Quat(
                self.x + t*(other_.x - self.x),
                self.y + t*(other_.y - self.y),
                self.z + t*(other_.z - self.z),
                self.w + t*(other_.w - self.w),
            )
            return result.normalized()

        theta_0 = math.acos(clamp(dot, -1.0, 1.0))
        theta   = theta_0 * t
        sin_t   = math.sin(theta)
        sin_0   = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_t / (sin_0 + EPSILON)
        s1 = sin_t / (sin_0 + EPSILON)

        return Quat(
            s0*self.x + s1*other_.x,
            s0*self.y + s1*other_.y,
            s0*self.z + s1*other_.z,
            s0*self.w + s1*other_.w,
        ).normalized()

    def to_euler(self) -> tuple[float, float, float]:
        """
        Конвертація у Euler кути (XYZ, радіани).
        Повертає (pitch, yaw, roll).
        """
        # Roll (X)
        sinr_cosp = 2.0 * (self.w*self.x + self.y*self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x**2 + self.y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (Y)
        sinp = 2.0 * (self.w*self.y - self.z*self.x)
        pitch = (math.copysign(HALF_PI, sinp)
                 if abs(sinp) >= 1.0
                 else math.asin(sinp))

        # Yaw (Z)
        siny_cosp = 2.0 * (self.w*self.z + self.x*self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y**2 + self.z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (pitch, yaw, roll)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.w]

    # ---- Конструктори ----
    @classmethod
    def identity(cls) -> "Quat":
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle_rad: float) -> "Quat":
        """Осьовий кут → кватерніон."""
        ax = axis.normalized()
        s  = math.sin(angle_rad * 0.5)
        c  = math.cos(angle_rad * 0.5)
        return cls(ax.x*s, ax.y*s, ax.z*s, c).normalized()

    @classmethod
    def from_euler(cls,
        pitch: float,    # rotation around X
        yaw:   float,    # rotation around Y
        roll:  float,    # rotation around Z
    ) -> "Quat":
        """Euler кути (радіани) → кватерніон."""
        cy = math.cos(yaw   * 0.5)
        sy = math.sin(yaw   * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll  * 0.5)
        sr = math.sin(roll  * 0.5)
        return cls(
            x = sr*cp*cy - cr*sp*sy,
            y = cr*sp*cy + sr*cp*sy,
            z = cr*cp*sy - sr*sp*cy,
            w = cr*cp*cy + sr*sp*sy,
        ).normalized()

    @classmethod
    def from_euler_deg(cls, pitch: float, yaw: float, roll: float) -> "Quat":
        """Euler кути у ГРАДУСАХ → кватерніон."""
        return cls.from_euler(
            pitch * DEG2RAD, yaw * DEG2RAD, roll * DEG2RAD
        )

    @classmethod
    def look_rotation(cls, forward: Vec3, up: Vec3 = Vec3(0,1,0)) -> "Quat":
        """
        Кватерніон що повертає об'єкт щоб він дивився у forward.
        Аналог Three.js Quaternion.setFromRotationMatrix(lookAt()).
        """
        f = forward.normalized()
        r = up.cross(f).normalized()
        if r.length_sq < EPSILON:
            r = Vec3(1.0, 0.0, 0.0)
        u = f.cross(r)

        # Rotation matrix → Quaternion (Shepperd method)
        trace = r.x + u.y + f.z

        if trace > 0.0:
            s = 0.5 / math.sqrt(trace + 1.0)
            return cls(
                (u.z - f.y) * s,
                (f.x - r.z) * s,
                (r.y - u.x) * s,
                0.25 / s,
            ).normalized()
        elif r.x > u.y and r.x > f.z:
            s = 2.0 * math.sqrt(1.0 + r.x - u.y - f.z)
            return cls(
                0.25 * s,
                (r.y + u.x) / s,
                (f.x + r.z) / s,
                (u.z - f.y) / s,
            ).normalized()
        elif u.y > f.z:
            s = 2.0 * math.sqrt(1.0 + u.y - r.x - f.z)
            return cls(
                (r.y + u.x) / s,
                0.25 * s,
                (u.z + f.y) / s,
                (f.x - r.z) / s,
            ).normalized()
        else:
            s = 2.0 * math.sqrt(1.0 + f.z - r.x - u.y)
            return cls(
                (f.x + r.z) / s,
                (u.z + f.y) / s,
                0.25 * s,
                (r.y - u.x) / s,
            ).normalized()

    def __repr__(self) -> str:
        return (f"Quat(x={self.x:.4f}, y={self.y:.4f}, "
                f"z={self.z:.4f}, w={self.w:.4f})")


# ────────────────────────────────────────────────────────────────
# MAT4
# ────────────────────────────────────────────────────────────────

class Mat4:
    """
    4×4 матриця трансформації (column-major, як у OpenGL/Three.js).

    Внутрішнє представлення: list[float] довжиною 16.
    Індексування: _m[col * 4 + row]

    Не frozen (мутабельна) для ефективності обчислень.
    Використовується лише всередині engine — назовні передаємо
    як tuple або list.
    """

    __slots__ = ("_m",)

    def __init__(self, elements: Sequence[float] | None = None) -> None:
        if elements is not None:
            assert len(elements) == 16
            self._m: list[float] = list(elements)
        else:
            # Identity
            self._m = [
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1,
            ]

    # ---- Доступ до елементів ----
    def get(self, row: int, col: int) -> float:
        return self._m[col * 4 + row]

    def set(self, row: int, col: int, value: float) -> None:
        self._m[col * 4 + row] = value

    # ---- Арифметика ----
    def __mul__(self, other: "Mat4 | Vec3 | Vec4") -> "Mat4 | Vec3 | Vec4":
        if isinstance(other, Mat4):
            return self._mul_mat(other)
        if isinstance(other, Vec4):
            return self._mul_vec4(other)
        if isinstance(other, Vec3):
            # Treat Vec3 as point (w=1)
            v4 = self._mul_vec4(Vec4(other.x, other.y, other.z, 1.0))
            w  = v4.w
            if abs(w) < EPSILON:
                return Vec3(v4.x, v4.y, v4.z)
            return Vec3(v4.x/w, v4.y/w, v4.z/w)
        return NotImplemented

    def _mul_mat(self, other: "Mat4") -> "Mat4":
        a, b = self._m, other._m
        result = [0.0] * 16
        for col in range(4):
            for row in range(4):
                s = 0.0
                for k in range(4):
                    s += a[k*4 + row] * b[col*4 + k]
                result[col*4 + row] = s
        return Mat4(result)

    def _mul_vec4(self, v: Vec4) -> Vec4:
        m = self._m
        return Vec4(
            m[0]*v.x + m[4]*v.y + m[ 8]*v.z + m[12]*v.w,
            m[1]*v.x + m[5]*v.y + m[ 9]*v.z + m[13]*v.w,
            m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
            m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]*v.w,
        )

    # ---- Transpose / Inverse ----
    def transposed(self) -> "Mat4":
        m = self._m
        return Mat4([
            m[0],m[4],m[ 8],m[12],
            m[1],m[5],m[ 9],m[13],
            m[2],m[6],m[10],m[14],
            m[3],m[7],m[11],m[15],
        ])

    def inverted(self) -> "Mat4":
        """
        Inverse матриці (Gauss-Jordan elimination).
        Для affine матриць (без шкалювання) ефективніше
        використовувати inverted_affine().
        """
        m  = self._m
        inv = [0.0] * 16

        inv[0]  = ( m[5]*m[10]*m[15] - m[5]*m[11]*m[14]
                  - m[9]*m[6]*m[15]  + m[9]*m[7]*m[14]
                  + m[13]*m[6]*m[11] - m[13]*m[7]*m[10])
        inv[4]  = (-m[4]*m[10]*m[15] + m[4]*m[11]*m[14]
                  + m[8]*m[6]*m[15]  - m[8]*m[7]*m[14]
                  - m[12]*m[6]*m[11] + m[12]*m[7]*m[10])
        inv[8]  = ( m[4]*m[9]*m[15]  - m[4]*m[11]*m[13]
                  - m[8]*m[5]*m[15]  + m[8]*m[7]*m[13]
                  + m[12]*m[5]*m[11] - m[12]*m[7]*m[9])
        inv[12] = (-m[4]*m[9]*m[14]  + m[4]*m[10]*m[13]
                  + m[8]*m[5]*m[14]  - m[8]*m[6]*m[13]
                  - m[12]*m[5]*m[10] + m[12]*m[6]*m[9])

        det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12]
        if abs(det) < EPSILON:
            return Mat4()   # identity як fallback

        inv[1]  = (-m[1]*m[10]*m[15] + m[1]*m[11]*m[14]
                  + m[9]*m[2]*m[15]  - m[9]*m[3]*m[14]
                  - m[13]*m[2]*m[11] + m[13]*m[3]*m[10])
        inv[5]  = ( m[0]*m[10]*m[15] - m[0]*m[11]*m[14]
                  - m[8]*m[2]*m[15]  + m[8]*m[3]*m[14]
                  + m[12]*m[2]*m[11] - m[12]*m[3]*m[10])
        inv[9]  = (-m[0]*m[9]*m[15]  + m[0]*m[11]*m[13]
                  + m[8]*m[1]*m[15]  - m[8]*m[3]*m[13]
                  - m[12]*m[1]*m[11] + m[12]*m[3]*m[9])
        inv[13] = ( m[0]*m[9]*m[14]  - m[0]*m[10]*m[13]
                  - m[8]*m[1]*m[14]  + m[8]*m[2]*m[13]
                  + m[12]*m[1]*m[10] - m[12]*m[2]*m[9])
        inv[2]  = ( m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]
                  - m[5]*m[2]*m[15]  + m[5]*m[3]*m[14]
                  + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6])
        inv[6]  = (-m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]
                  + m[4]*m[2]*m[15]  - m[4]*m[3]*m[14]
                  - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6])
        inv[10] = ( m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]
                  - m[4]*m[1]*m[15]  + m[4]*m[3]*m[13]
                  + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5])
        inv[14] = (-m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]
                  + m[4]*m[1]*m[14]  - m[4]*m[2]*m[13]
                  - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5])
        inv[3]  = (-m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]
                  + m[5]*m[2]*m[11]  - m[5]*m[3]*m[10]
                  - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6])
        inv[7]  = ( m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]
                  - m[4]*m[2]*m[11]  + m[4]*m[3]*m[10]
                  + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6])
        inv[11] = (-m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]
                  + m[4]*m[1]*m[11]  - m[4]*m[3]*m[9]
                  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5])
        inv[15] = ( m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]
                  - m[4]*m[1]*m[10]  + m[4]*m[2]*m[9]
                  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5])

        inv_det = 1.0 / det
        return Mat4([x * inv_det for x in inv])

    def inverted_affine(self) -> "Mat4":
        """
        Inverse для affine матриці (тільки rotation + translation).
        Набагато швидше ніж повний inverse.
        """
        m = self._m
        # Transpose rotation part
        r00, r01, r02 = m[0], m[1], m[2]
        r10, r11, r12 = m[4], m[5], m[6]
        r20, r21, r22 = m[8], m[9], m[10]
        tx, ty, tz = m[12], m[13], m[14]

        # Inverse translation = -R^T * t
        itx = -(r00*tx + r01*ty + r02*tz)
        ity = -(r10*tx + r11*ty + r12*tz)
        itz = -(r20*tx + r21*ty + r22*tz)

        return Mat4([
            r00, r10, r20, 0,
            r01, r11, r21, 0,
            r02, r12, r22, 0,
            itx, ity, itz, 1,
        ])

    # ---- Статичні конструктори ----
    @classmethod
    def identity(cls) -> "Mat4":
        return cls()

    @classmethod
    def translation(cls, tx: float, ty: float, tz: float) -> "Mat4":
        return cls([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            tx, ty, tz, 1,
        ])

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "Mat4":
        return cls([
            sx,  0,  0, 0,
             0, sy,  0, 0,
             0,  0, sz, 0,
             0,  0,  0, 1,
        ])

    @classmethod
    def rotation_x(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1])

    @classmethod
    def rotation_y(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1])

    @classmethod
    def rotation_z(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1])

    @classmethod
    def from_quat(cls, q: Quat) -> "Mat4":
        """Кватерніон → rotation matrix."""
        qn = q.normalized()
        x, y, z, w = qn.x, qn.y, qn.z, qn.w
        return cls([
            1-2*(y*y+z*z),   2*(x*y+w*z),   2*(x*z-w*y), 0,
              2*(x*y-w*z), 1-2*(x*x+z*z),   2*(y*z+w*x), 0,
              2*(x*z+w*y),   2*(y*z-w*x), 1-2*(x*x+y*y), 0,
            0,               0,               0,           1,
        ])

    @classmethod
    def look_at(cls,
        eye:    Vec3,
        target: Vec3,
        up:     Vec3 = Vec3(0, 1, 0),
    ) -> "Mat4":
        """
        View matrix: камера дивиться з eye на target.
        Аналог glm::lookAt.
        """
        f = (eye - target).normalized()
        r = up.cross(f).normalized()
        if r.length_sq < EPSILON:
            r = Vec3(1.0, 0.0, 0.0)
        u = f.cross(r)

        return cls([
            r.x,  u.x,  f.x, 0,
            r.y,  u.y,  f.y, 0,
            r.z,  u.z,  f.z, 0,
            -r.dot(eye), -u.dot(eye), -f.dot(eye), 1,
        ])

    @classmethod
    def perspective(cls,
        fov_y:  float,    # vertical FOV у радіанах
        aspect: float,    # width / height
        near:   float,
        far:    float,
    ) -> "Mat4":
        """Perspective projection matrix (OpenGL convention)."""
        f = 1.0 / math.tan(fov_y * 0.5)
        nf = 1.0 / (near - far)
        return cls([
            f/aspect, 0,              0,  0,
            0,        f,              0,  0,
            0,        0,  (far+near)*nf, -1,
            0,        0, 2*far*near*nf,   0,
        ])

    @classmethod
    def orthographic(cls,
        left: float, right: float,
        bottom: float, top: float,
        near: float, far: float,
    ) -> "Mat4":
        """Orthographic projection matrix."""
        rl = 1.0 / (right - left)
        tb = 1.0 / (top   - bottom)
        fn = 1.0 / (far   - near)
        return cls([
            2*rl, 0,    0,    0,
            0,    2*tb, 0,    0,
            0,    0,   -2*fn, 0,
            -(right+left)*rl, -(top+bottom)*tb, -(far+near)*fn, 1,
        ])

    # ---- Утиліти ----
    def to_list(self) -> list[float]:
        return list(self._m)

    def to_column_major(self) -> list[float]:
        """Column-major для OpenGL / WebGL."""
        return list(self._m)

    def to_row_major(self) -> list[float]:
        """Row-major для DirectX."""
        return self.transposed()._m

    @classmethod
    def from_list(cls, data: list[float]) -> "Mat4":
        return cls(data)

    def __repr__(self) -> str:
        m = self._m
        rows = [
            f"  [{m[0]:8.4f} {m[4]:8.4f} {m[ 8]:8.4f} {m[12]:8.4f}]",
            f"  [{m[1]:8.4f} {m[5]:8.4f} {m[ 9]:8.4f} {m[13]:8.4f}]",
            f"  [{m[2]:8.4f} {m[6]:8.4f} {m[10]:8.4f} {m[14]:8.4f}]",
            f"  [{m[3]:8.4f} {m[7]:8.4f} {m[11]:8.4f} {m[15]:8.4f}]",
        ]
        return "Mat4(\n" + "\n".join(rows) + "\n)"


# ────────────────────────────────────────────────────────────────
# AABB (Axis-Aligned Bounding Box)
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AABB:
    """
    Вісь-вирівняний обмежуючий паралелепіпед.
    Використовується для frustum culling та collision detection.
    """
    min: Vec3 = field(default_factory=Vec3.zero)
    max: Vec3 = field(default_factory=Vec3.zero)

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )

    @property
    def size(self) -> Vec3:
        return self.max - self.min

    @property
    def half_size(self) -> Vec3:
        return self.size * 0.5

    @property
    def is_valid(self) -> bool:
        return (self.min.x <= self.max.x and
                self.min.y <= self.max.y and
                self.min.z <= self.max.z)

    def contains(self, point: Vec3) -> bool:
        return (self.min.x <= point.x <= self.max.x and
                self.min.y <= point.y <= self.max.y and
                self.min.z <= point.z <= self.max.z)

    def intersects(self, other: "AABB") -> bool:
        return (self.min.x <= other.max.x and self.max.x >= other.min.x and
                self.min.y <= other.max.y and self.max.y >= other.min.y and
                self.min.z <= other.max.z and self.max.z >= other.min.z)

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            min=Vec3(
                min(self.min.x, other.min.x),
                min(self.min.y, other.min.y),
                min(self.min.z, other.min.z),
            ),
            max=Vec3(
                max(self.max.x, other.max.x),
                max(self.max.y, other.max.y),
                max(self.max.z, other.max.z),
            ),
        )

    def expand(self, amount: float) -> "AABB":
        v = Vec3(amount, amount, amount)
        return AABB(self.min - v, self.max + v)

    def transform(self, mat: Mat4) -> "AABB":
        """Трансформувати AABB матрицею (результат може бути більшим)."""
        corners = [
            Vec3(self.min.x, self.min.y, self.min.z),
            Vec3(self.max.x, self.min.y, self.min.z),
            Vec3(self.min.x, self.max.y, self.min.z),
            Vec3(self.max.x, self.max.y, self.min.z),
            Vec3(self.min.x, self.min.y, self.max.z),
            Vec3(self.max.x, self.min.y, self.max.z),
            Vec3(self.min.x, self.max.y, self.max.z),
            Vec3(self.max.x, self.max.y, self.max.z),
        ]
        transformed = [mat * c for c in corners]
        return AABB.from_points(transformed)

    @classmethod
    def from_points(cls, points: list[Vec3]) -> "AABB":
        if not points:
            return cls()
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        return cls(
            min=Vec3(min(xs), min(ys), min(zs)),
            max=Vec3(max(xs), max(ys), max(zs)),
        )

    @classmethod
    def from_center_size(cls, center: Vec3, size: Vec3) -> "AABB":
        half = size * 0.5
        return cls(center - half, center + half)

    def __repr__(self) -> str:
        return f"AABB(min={self.min}, max={self.max})"


# ────────────────────────────────────────────────────────────────
# RAY
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Ray:
    """
    Промінь у 3D просторі.
    Використовується для ray casting, mouse picking, line-of-sight.
    """
    origin:    Vec3
    direction: Vec3   # має бути нормалізований

    @classmethod
    def from_points(cls, start: Vec3, end: Vec3) -> "Ray":
        return cls(origin=start, direction=(end - start).normalized())

    def at(self, t: float) -> Vec3:
        """Точка на промені при параметрі t."""
        return self.origin + self.direction * t

    def intersect_aabb(self, aabb: AABB) -> float | None:
        """
        Перетин Ray–AABB (slab method).
        Повертає t (відстань) або None якщо немає перетину.
        """
        t_min = -math.inf
        t_max =  math.inf

        for axis in range(3):
            o = (self.origin.x, self.origin.y, self.origin.z)[axis]
            d = (self.direction.x, self.direction.y, self.direction.z)[axis]
            lo = (aabb.min.x, aabb.min.y, aabb.min.z)[axis]
            hi = (aabb.max.x, aabb.max.y, aabb.max.z)[axis]

            if abs(d) < EPSILON:
                if o < lo or o > hi:
                    return None
            else:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None

        return t_min if t_min >= 0.0 else (t_max if t_max >= 0.0 else None)

    def intersect_triangle(
        self,
        v0: Vec3, v1: Vec3, v2: Vec3,
    ) -> float | None:
        """
        Möller–Trumbore алгоритм перетину Ray–Triangle.
        Повертає t або None.
        """
        edge1 = v1 - v0
        edge2 = v2 - v0
        h     = self.direction.cross(edge2)
        a     = edge1.dot(h)

        if abs(a) < EPSILON:
            return None   # Ray паралельний трикутнику

        f  = 1.0 / a
        s  = self.origin - v0
        u  = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            return None

        q = s.cross(edge1)
        v = f * self.direction.dot(q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * edge2.dot(q)
        return t if t > EPSILON else None

    def intersect_sphere(
        self,
        center: Vec3,
        radius: float,
    ) -> float | None:
        """Перетин Ray–Sphere. Повертає найближчий t або None."""
        oc = self.origin - center
        a  = self.direction.dot(self.direction)
        b  = 2.0 * oc.dot(self.direction)
        c  = oc.dot(oc) - radius * radius
        d  = b * b - 4 * a * c

        if d < 0.0:
            return None

        sqrt_d = math.sqrt(d)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t1 > EPSILON:
            return t1
        if t2 > EPSILON:
            return t2
        return None

    def closest_point(self, point: Vec3) -> Vec3:
        """Найближча точка на промені до заданої точки."""
        t = max(0.0, (point - self.origin).dot(self.direction))
        return self.at(t)

    def distance_to_point(self, point: Vec3) -> float:
        return (self.closest_point(point) - point).length

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, direction={self.direction})"
