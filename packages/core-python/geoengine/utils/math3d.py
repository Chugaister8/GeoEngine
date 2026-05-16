"""
GeoEngine — 3D Math Utilities
Вектори, матриці, кватерніони — без зовнішніх залежностей.

Всі операції — pure Python + numpy для batch обчислень.
Сумісні з Three.js конвенцією (Y вгору, right-handed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import numpy.typing as npt


# ----------------------------------------------------------------
# VEC2
# ----------------------------------------------------------------

@dataclass(slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return f"Vec2({self.x:.4f}, {self.y:.4f})"

    @property
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        l = self.length
        if l < 1e-10:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / l, self.y / l)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def lerp(self, other: "Vec2", t: float) -> "Vec2":
        return Vec2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> "Vec2":
        return cls(t[0], t[1])

    @classmethod
    def zero(cls) -> "Vec2":
        return cls(0.0, 0.0)

    @classmethod
    def one(cls) -> "Vec2":
        return cls(1.0, 1.0)


# ----------------------------------------------------------------
# VEC3
# ----------------------------------------------------------------

@dataclass(slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

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

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    @property
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def length_sq(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    def normalized(self) -> "Vec3":
        l = self.length
        if l < 1e-10:
            return Vec3(0.0, 1.0, 0.0)  # default up
        return Vec3(self.x / l, self.y / l, self.z / l)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).length

    def angle_to(self, other: "Vec3") -> float:
        """Кут між двома векторами (радіани)."""
        cos_a = self.dot(other) / max(self.length * other.length, 1e-10)
        return math.acos(max(-1.0, min(1.0, cos_a)))

    def reflect(self, normal: "Vec3") -> "Vec3":
        """Відбиття відносно нормалі."""
        n = normal.normalized()
        return self - n * (2.0 * self.dot(n))

    def project_onto(self, other: "Vec3") -> "Vec3":
        """Проєкція на вектор other."""
        other_len_sq = other.length_sq
        if other_len_sq < 1e-10:
            return Vec3.zero()
        return other * (self.dot(other) / other_len_sq)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_numpy(self) -> npt.NDArray[np.float32]:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float]) -> "Vec3":
        return cls(t[0], t[1], t[2])

    @classmethod
    def from_numpy(cls, arr: npt.NDArray) -> "Vec3":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

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
        return cls(0.0, 0.0, -1.0)   # Three.js: -Z вперед

    @classmethod
    def right(cls) -> "Vec3":
        return cls(1.0, 0.0, 0.0)


# ----------------------------------------------------------------
# VEC4
# ----------------------------------------------------------------

@dataclass(slots=True)
class Vec4:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z
        yield self.w

    def to_vec3(self) -> Vec3:
        if abs(self.w) < 1e-10:
            return Vec3(self.x, self.y, self.z)
        return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    def to_numpy(self) -> npt.NDArray[np.float32]:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)


# ----------------------------------------------------------------
# QUAT (Quaternion)
# ----------------------------------------------------------------

@dataclass(slots=True)
class Quat:
    """
    Quaternion для представлення обертань.
    Конвенція: (x, y, z, w) де w = cos(angle/2).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def identity(cls) -> "Quat":
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle_rad: float) -> "Quat":
        """Кватерніон з осі обертання та кута."""
        axis = axis.normalized()
        half = angle_rad * 0.5
        s    = math.sin(half)
        return cls(
            x=axis.x * s,
            y=axis.y * s,
            z=axis.z * s,
            w=math.cos(half),
        )

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> "Quat":
        """
        Euler кути (радіани) → Quaternion.
        Порядок: Y (yaw) → X (pitch) → Z (roll)
        """
        cy = math.cos(yaw   * 0.5)
        sy = math.sin(yaw   * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll  * 0.5)
        sr = math.sin(roll  * 0.5)
        return cls(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )

    def __mul__(self, other: "Quat") -> "Quat":
        """Множення кватерніонів (композиція обертань)."""
        return Quat(
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    def rotate_vec3(self, v: Vec3) -> Vec3:
        """Повернути вектор за цим кватерніоном."""
        # q * v * q^-1 (оптимізована формула)
        qv = Vec3(self.x, self.y, self.z)
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * (2.0 * self.w)) + (uuv * 2.0)

    def conjugate(self) -> "Quat":
        return Quat(-self.x, -self.y, -self.z, self.w)

    def normalized(self) -> "Quat":
        n = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if n < 1e-10:
            return Quat.identity()
        return Quat(self.x/n, self.y/n, self.z/n, self.w/n)

    def slerp(self, other: "Quat", t: float) -> "Quat":
        """Spherical Linear Interpolation між двома кватерніонами."""
        dot = (self.x * other.x + self.y * other.y
               + self.z * other.z + self.w * other.w)

        # Якщо dot від'ємний — інвертуємо один кватерніон
        q2 = other if dot >= 0 else Quat(-other.x, -other.y, -other.z, -other.w)
        dot = abs(dot)

        if dot > 0.9995:
            # Майже однакові — лінійна інтерполяція
            result = Quat(
                self.x + t * (q2.x - self.x),
                self.y + t * (q2.y - self.y),
                self.z + t * (q2.z - self.z),
                self.w + t * (q2.w - self.w),
            )
            return result.normalized()

        theta_0   = math.acos(dot)
        theta     = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return Quat(
            s1 * self.x + s2 * q2.x,
            s1 * self.y + s2 * q2.y,
            s1 * self.z + s2 * q2.z,
            s1 * self.w + s2 * q2.w,
        )

    def to_euler(self) -> tuple[float, float, float]:
        """Кватерніон → Euler кути (pitch, yaw, roll) у радіанах."""
        # Pitch (X)
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Yaw (Y)
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        # Roll (Z)
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return pitch, yaw, roll

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


# ----------------------------------------------------------------
# MAT4 (4×4 матриця, column-major як у WebGPU/Three.js)
# ----------------------------------------------------------------

class Mat4:
    """
    4×4 матриця трансформацій (column-major).
    Сумісна з Three.js Matrix4 та WebGPU uniform буферами.

    Зберігається як flat list[float] довжиною 16:
      [m00, m10, m20, m30,   ← колонка 0
       m01, m11, m21, m31,   ← колонка 1
       m02, m12, m22, m32,   ← колонка 2
       m03, m13, m23, m33]   ← колонка 3
    """

    __slots__ = ("_m",)

    def __init__(self, m: list[float] | None = None) -> None:
        if m is not None:
            if len(m) != 16:
                raise ValueError(f"Mat4 потребує 16 елементів, отримано {len(m)}")
            self._m = list(m)
        else:
            # Identity матриця
            self._m = [
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1,
            ]

    # ---- Фабрики ----

    @classmethod
    def identity(cls) -> "Mat4":
        return cls()

    @classmethod
    def translation(cls, x: float, y: float, z: float) -> "Mat4":
        m = cls()
        m._m[12] = x
        m._m[13] = y
        m._m[14] = z
        return m

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "Mat4":
        m = cls()
        m._m[0]  = sx
        m._m[5]  = sy
        m._m[10] = sz
        return m

    @classmethod
    def rotation_x(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([
            1, 0,  0, 0,
            0, c,  s, 0,
            0, -s, c, 0,
            0, 0,  0, 1,
        ])

    @classmethod
    def rotation_y(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([
            c, 0, -s, 0,
            0, 1,  0, 0,
            s, 0,  c, 0,
            0, 0,  0, 1,
        ])

    @classmethod
    def rotation_z(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([
            c,  s, 0, 0,
            -s, c, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1,
        ])

    @classmethod
    def from_quat(cls, q: Quat) -> "Mat4":
        """Quaternion → rotation Mat4."""
        x2 = q.x * 2; y2 = q.y * 2; z2 = q.z * 2
        xx = q.x * x2; yy = q.y * y2; zz = q.z * z2
        xy = q.x * y2; xz = q.x * z2; yz = q.y * z2
        wx = q.w * x2; wy = q.w * y2; wz = q.w * z2
        return cls([
            1-(yy+zz), xy+wz,    xz-wy,    0,
            xy-wz,     1-(xx+zz), yz+wx,    0,
            xz+wy,     yz-wx,    1-(xx+yy), 0,
            0,         0,         0,         1,
        ])

    @classmethod
    def look_at(cls, eye: Vec3, target: Vec3, up: Vec3) -> "Mat4":
        """View матриця (LookAt)."""
        f = (target - eye).normalized()
        r = f.cross(up).normalized()
        u = r.cross(f)
        return cls([
            r.x,  u.x,  -f.x, 0,
            r.y,  u.y,  -f.y, 0,
            r.z,  u.z,  -f.z, 0,
            -r.dot(eye), -u.dot(eye), f.dot(eye), 1,
        ])

    @classmethod
    def perspective(
        cls,
        fov_y: float,    # радіани
        aspect: float,
        near:   float,
        far:    float,
    ) -> "Mat4":
        """Perspective projection матриця (WebGPU NDC)."""
        f   = 1.0 / math.tan(fov_y * 0.5)
        nf  = 1.0 / (near - far)
        return cls([
            f / aspect, 0, 0,               0,
            0,          f, 0,               0,
            0,          0, (far + near) * nf, -1,
            0,          0, 2 * far * near * nf, 0,
        ])

    @classmethod
    def orthographic(
        cls,
        left: float, right: float,
        bottom: float, top: float,
        near: float, far: float,
    ) -> "Mat4":
        """Orthographic projection матриця."""
        lr = 1.0 / (left - right)
        bt = 1.0 / (bottom - top)
        nf = 1.0 / (near - far)
        return cls([
            -2 * lr,   0,         0,         0,
             0,        -2 * bt,   0,         0,
             0,         0,         2 * nf,    0,
            (left+right)*lr, (top+bottom)*bt, (far+near)*nf, 1,
        ])

    # ---- Операції ----

    def __mul__(self, other: "Mat4") -> "Mat4":
        """Множення матриць."""
        a, b = self._m, other._m
        return Mat4([
            a[0]*b[0]  + a[4]*b[1]  + a[8]*b[2]  + a[12]*b[3],
            a[1]*b[0]  + a[5]*b[1]  + a[9]*b[2]  + a[13]*b[3],
            a[2]*b[0]  + a[6]*b[1]  + a[10]*b[2] + a[14]*b[3],
            a[3]*b[0]  + a[7]*b[1]  + a[11]*b[2] + a[15]*b[3],
            a[0]*b[4]  + a[4]*b[5]  + a[8]*b[6]  + a[12]*b[7],
            a[1]*b[4]  + a[5]*b[5]  + a[9]*b[6]  + a[13]*b[7],
            a[2]*b[4]  + a[6]*b[5]  + a[10]*b[6] + a[14]*b[7],
            a[3]*b[4]  + a[7]*b[5]  + a[11]*b[6] + a[15]*b[7],
            a[0]*b[8]  + a[4]*b[9]  + a[8]*b[10] + a[12]*b[11],
            a[1]*b[8]  + a[5]*b[9]  + a[9]*b[10] + a[13]*b[11],
            a[2]*b[8]  + a[6]*b[9]  + a[10]*b[10]+ a[14]*b[11],
            a[3]*b[8]  + a[7]*b[9]  + a[11]*b[10]+ a[15]*b[11],
            a[0]*b[12] + a[4]*b[13] + a[8]*b[14] + a[12]*b[15],
            a[1]*b[12] + a[5]*b[13] + a[9]*b[14] + a[13]*b[15],
            a[2]*b[12] + a[6]*b[13] + a[10]*b[14]+ a[14]*b[15],
            a[3]*b[12] + a[7]*b[13] + a[11]*b[14]+ a[15]*b[15],
        ])

    def transform_vec3(self, v: Vec3, w: float = 1.0) -> Vec3:
        """Трансформувати Vec3 цією матрицею (w=1 для точок, w=0 для векторів)."""
        m = self._m
        x = m[0]*v.x + m[4]*v.y + m[8]*v.z  + m[12]*w
        y = m[1]*v.x + m[5]*v.y + m[9]*v.z  + m[13]*w
        z = m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*w
        rw= m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]*w
        if abs(rw) > 1e-10 and w != 0.0:
            return Vec3(x/rw, y/rw, z/rw)
        return Vec3(x, y, z)

    def transposed(self) -> "Mat4":
        m = self._m
        return Mat4([
            m[0], m[4], m[8],  m[12],
            m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15],
        ])

    def inverted(self) -> "Mat4":
        """Обернена матриця (Gauss-Jordan)."""
        arr = np.array(self._m, dtype=np.float64).reshape(4, 4)
        try:
            inv = np.linalg.inv(arr)
            return Mat4(inv.flatten().tolist())
        except np.linalg.LinAlgError as exc:
            raise ValueError("Матриця не оборотна") from exc

    def to_numpy(self) -> npt.NDArray[np.float32]:
        return np.array(self._m, dtype=np.float32)

    def to_list(self) -> list[float]:
        return list(self._m)

    def __repr__(self) -> str:
        m = self._m
        rows = []
        for r in range(4):
            row = [m[r + c * 4] for c in range(4)]
            rows.append("[" + ", ".join(f"{v:8.4f}" for v in row) + "]")
        return "Mat4(\n  " + "\n  ".join(rows) + "\n)"


# ----------------------------------------------------------------
# AABB (Axis-Aligned Bounding Box)
# ----------------------------------------------------------------

@dataclass(slots=True)
class AABB:
    """Axis-Aligned Bounding Box у 3D просторі."""
    min: Vec3
    max: Vec3

    @classmethod
    def from_points(cls, points: list[Vec3]) -> "AABB":
        if not points:
            return cls(Vec3.zero(), Vec3.zero())
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        return cls(
            min=Vec3(min(xs), min(ys), min(zs)),
            max=Vec3(max(xs), max(ys), max(zs)),
        )

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

    def contains(self, point: Vec3) -> bool:
        return (
            self.min.x <= point.x <= self.max.x and
            self.min.y <= point.y <= self.max.y and
            self.min.z <= point.z <= self.max.z
        )

    def intersects(self, other: "AABB") -> bool:
        return (
            self.min.x <= other.max.x and self.max.x >= other.min.x and
            self.min.y <= other.max.y and self.max.y >= other.min.y and
            self.min.z <= other.max.z and self.max.z >= other.min.z
        )

    def expanded(self, amount: float) -> "AABB":
        v = Vec3(amount, amount, amount)
        return AABB(self.min - v, self.max + v)

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


# ----------------------------------------------------------------
# RAY
# ----------------------------------------------------------------

@dataclass(slots=True)
class Ray:
    """Промінь у 3D просторі."""
    origin:    Vec3
    direction: Vec3   # normalized

    def __post_init__(self) -> None:
        self.direction = self.direction.normalized()

    def at(self, t: float) -> Vec3:
        """Точка на промені при параметрі t."""
        return self.origin + self.direction * t

    def intersect_aabb(self, aabb: AABB) -> float | None:
        """
        Перетин Ray з AABB.
        Returns: t (відстань вздовж ray) або None якщо немає перетину.
        """
        t_min = 0.0
        t_max = float("inf")

        for i, (attr_o, attr_d, attr_min, attr_max) in enumerate([
            ("x", "x", aabb.min.x, aabb.max.x),
            ("y", "y", aabb.min.y, aabb.max.y),
            ("z", "z", aabb.min.z, aabb.max.z),
        ]):
            o = getattr(self.origin,    attr_o)
            d = getattr(self.direction, attr_d)

            if abs(d) < 1e-10:
                if o < attr_min or o > attr_max:  # type: ignore[operator]
                    return None
                continue

            t1 = (attr_min - o) / d   # type: ignore[operator]
            t2 = (attr_max - o) / d   # type: ignore[operator]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return None

        return t_min if t_min >= 0 else None

    def intersect_triangle(
        self,
        v0: Vec3, v1: Vec3, v2: Vec3,
    ) -> float | None:
        """
        Möller–Trumbore алгоритм перетину ray-triangle.
        Returns: t (відстань) або None якщо немає перетину.
        """
        EPSILON = 1e-7
        edge1 = v1 - v0
        edge2 = v2 - v0
        h     = self.direction.cross(edge2)
        a     = edge1.dot(h)

        if abs(a) < EPSILON:
            return None   # паралельний

        f = 1.0 / a
        s = self.origin - v0
        u = f * s.dot(h)

        if not (0.0 <= u <= 1.0):
            return None

        q = s.cross(edge1)
        v = f * self.direction.dot(q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * edge2.dot(q)
        return t if t > EPSILON else None


# ----------------------------------------------------------------
# УТИЛІТИ
# ----------------------------------------------------------------

def lerp(a: float, b: float, t: float) -> float:
    """Лінійна інтерполяція."""
    return a + (b - a) * t

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Обмеження значення в діапазоні."""
    return max(min_val, min(max_val, value))

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep [0..1]."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def ease_in_out_cubic(t: float) -> float:
    """Кубічна easing функція."""
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2)**3 / 2

def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi

def angle_wrap_360(angle_deg: float) -> float:
    """Привести кут до діапазону [0, 360)."""
    return angle_deg % 360.0

def angle_wrap_180(angle_deg: float) -> float:
    """Привести кут до діапазону (-180, 180]."""
    angle = angle_deg % 360.0
    return angle - 360.0 if angle > 180.0 else angle
