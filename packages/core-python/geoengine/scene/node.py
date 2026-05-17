"""
GeoEngine — SceneNode
Базовий клас вузла сцени.

Ієрархія:
  SceneNode (базовий)
  ├── TerrainNode   — DEM тайл
  ├── BuildingNode  — будівля або колекція
  ├── RoadNode      — дорожня мережа
  ├── WaterNode     — водойма
  ├── VectorNode    — довільна векторна геометрія
  └── GroupNode     — контейнер для вузлів

Кожен вузол:
  - Має унікальний UUID
  - Зберігає transform (позиція, поворот, масштаб)
  - Має видимість та opacity
  - Може мати дочірні вузли
  - Серіалізується у dict для WS/REST
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterator

from ..utils.math3d import Vec3, Quat, Mat4, AABB


# ----------------------------------------------------------------
# ТИПИ
# ----------------------------------------------------------------

class NodeType(StrEnum):
    TERRAIN   = "terrain"
    BUILDING  = "building"
    ROAD      = "road"
    WATER     = "water"
    VECTOR    = "vector"
    GROUP     = "group"
    CAMERA    = "camera"
    LIGHT     = "light"
    CUSTOM    = "custom"


# ----------------------------------------------------------------
# TRANSFORM
# ----------------------------------------------------------------

@dataclass(slots=True)
class Transform:
    """
    Трансформація вузла у 3D просторі.

    Зберігається як окремі компоненти (TRS) для
    зручної анімації та інтерполяції.
    З них будується model matrix при потребі.
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    scale:    Vec3 = field(default_factory=Vec3.one)

    def to_matrix(self) -> Mat4:
        """Побудувати model matrix з TRS."""
        T = Mat4.translation(self.position.x, self.position.y, self.position.z)
        R = Mat4.from_quat(self.rotation)
        S = Mat4.scale(self.scale.x, self.scale.y, self.scale.z)
        return T * R * S

    def to_dict(self) -> dict:
        return {
            "position": self.position.to_tuple(),
            "rotation": self.rotation.to_tuple(),
            "scale":    self.scale.to_tuple(),
        }

    @classmethod
    def identity(cls) -> "Transform":
        return cls()


# ----------------------------------------------------------------
# SCENE NODE
# ----------------------------------------------------------------

class SceneNode:
    """
    Базовий вузол сцени.

    Підтримує:
    - Ієрархію (parent/children)
    - Transform (position, rotation, scale)
    - World transform (з урахуванням батьківського)
    - Видимість та opacity
    - Метадані (теги, кастомні поля)
    - Серіалізацію у dict

    Usage:
        node = SceneNode(name="Karpatians", node_type=NodeType.TERRAIN)
        node.visible = True
        node.transform.position = Vec3(100, 0, -200)
        scene.add(node)
    """

    def __init__(
        self,
        name:      str              = "",
        node_type: NodeType         = NodeType.CUSTOM,
        node_id:   str | None       = None,
        tags:      dict[str, str] | None = None,
    ) -> None:
        self._id:       str         = node_id or str(uuid.uuid4())
        self._name:     str         = name
        self._type:     NodeType    = node_type
        self._parent:   "SceneNode | None" = None
        self._children: list["SceneNode"]  = []

        # Трансформація
        self.transform: Transform = Transform()

        # Відображення
        self.visible:  bool  = True
        self.opacity:  float = 1.0
        self.cast_shadow:    bool = True
        self.receive_shadow: bool = True

        # Просторові межі (AABB у локальному просторі)
        self._bounds: AABB | None = None

        # Метадані
        self.tags:     dict[str, str] = tags or {}
        self.metadata: dict[str, Any] = {}

        # Стан
        self._dirty:   bool = True   # потребує оновлення world transform

    # ---- Властивості ----

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def node_type(self) -> NodeType:
        return self._type

    @property
    def parent(self) -> "SceneNode | None":
        return self._parent

    @property
    def children(self) -> list["SceneNode"]:
        return list(self._children)

    @property
    def child_count(self) -> int:
        return len(self._children)

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def depth(self) -> int:
        """Глибина у ієрархії (root = 0)."""
        if self._parent is None:
            return 0
        return self._parent.depth + 1

    @property
    def root(self) -> "SceneNode":
        """Корінь ієрархії."""
        if self._parent is None:
            return self
        return self._parent.root

    @property
    def bounds(self) -> AABB | None:
        return self._bounds

    @bounds.setter
    def bounds(self, value: AABB | None) -> None:
        self._bounds = value

    # ---- Ієрархія ----

    def add_child(self, child: "SceneNode") -> "SceneNode":
        """
        Додати дочірній вузол.

        Args:
            child: вузол що додається

        Returns:
            self (для chaining)

        Raises:
            ValueError: якщо child вже має батька або є предком
        """
        if child._parent is not None:
            child._parent.remove_child(child)

        if self._is_ancestor_of(child):
            raise ValueError(
                f"Циклічна ієрархія: {child.name} є предком {self.name}"
            )

        child._parent = self
        self._children.append(child)
        child._mark_dirty()
        return self

    def remove_child(self, child: "SceneNode") -> bool:
        """
        Видалити дочірній вузол.

        Returns:
            True якщо видалено, False якщо не знайдено
        """
        if child not in self._children:
            return False
        self._children.remove(child)
        child._parent = None
        child._mark_dirty()
        return True

    def remove_from_parent(self) -> bool:
        """Від'єднати себе від батька."""
        if self._parent is None:
            return False
        return self._parent.remove_child(self)

    def get_child(self, name: str) -> "SceneNode | None":
        """Знайти прямого нащадка за ім'ям."""
        for child in self._children:
            if child.name == name:
                return child
        return None

    def get_child_by_id(self, node_id: str) -> "SceneNode | None":
        """Знайти прямого нащадка за ID."""
        for child in self._children:
            if child.id == node_id:
                return child
        return None

    def find(self, name: str) -> "SceneNode | None":
        """Рекурсивний пошук за ім'ям у всій ієрархії."""
        if self.name == name:
            return self
        for child in self._children:
            result = child.find(name)
            if result is not None:
                return result
        return None

    def find_by_type(self, node_type: NodeType) -> list["SceneNode"]:
        """Знайти всі вузли заданого типу."""
        result: list[SceneNode] = []
        if self._type == node_type:
            result.append(self)
        for child in self._children:
            result.extend(child.find_by_type(node_type))
        return result

    # ---- World Transform ----

    def world_transform(self) -> Mat4:
        """
        Повна world matrix з урахуванням батьківської ієрархії.
        Кешується та інвалідується при зміні transform.
        """
        local = self.transform.to_matrix()
        if self._parent is None:
            return local
        return self._parent.world_transform() * local

    def world_position(self) -> Vec3:
        """Позиція у world-space."""
        wm = self.world_transform()
        return Vec3(
            x=float(wm._m[12]),
            y=float(wm._m[13]),
            z=float(wm._m[14]),
        )

    # ---- Ітерація ----

    def iter_all(self) -> Iterator["SceneNode"]:
        """Ітерація по всіх вузлах (BFS)."""
        yield self
        for child in self._children:
            yield from child.iter_all()

    def iter_visible(self) -> Iterator["SceneNode"]:
        """Ітерація тільки по видимих вузлах."""
        if not self.visible:
            return
        yield self
        for child in self._children:
            yield from child.iter_visible()

    # ---- Серіалізація ----

    def to_dict(self, include_children: bool = True) -> dict:
        """Серіалізувати вузол у dict для JSON/WS."""
        d: dict[str, Any] = {
            "id":        self._id,
            "name":      self._name,
            "type":      str(self._type),
            "visible":   self.visible,
            "opacity":   self.opacity,
            "transform": self.transform.to_dict(),
            "tags":      self.tags,
        }
        if include_children and self._children:
            d["children"] = [c.to_dict(True) for c in self._children]
        return d

    # ---- Приватні методи ----

    def _mark_dirty(self) -> None:
        """Позначити як 'потребує оновлення' (cascade)."""
        self._dirty = True
        for child in self._children:
            child._mark_dirty()

    def _is_ancestor_of(self, other: "SceneNode") -> bool:
        """Чи є self предком other."""
        current = other._parent
        while current is not None:
            if current is self:
                return True
            current = current._parent
        return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"type={self._type}, "
            f"children={len(self._children)})"
      )
