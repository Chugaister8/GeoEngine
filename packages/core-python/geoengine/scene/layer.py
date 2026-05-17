"""
GeoEngine — Layer System
Шари для організації контенту сцени.

Шар = NamedGroup + visibility + opacity + order.
Аналог шарів у GIS (QGIS, ArcGIS) або Photoshop.

Типові шари:
  base_terrain  — DEM терейн (завжди знизу)
  satellite     — супутникові знімки
  osm_buildings — будівлі OSM
  osm_roads     — дороги OSM
  analysis      — аналітичні overlay (slope, viewshed)
  custom_*      — довільні користувацькі шари
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .node import SceneNode, NodeType


# ----------------------------------------------------------------
# LAYER
# ----------------------------------------------------------------

@dataclass
class Layer:
    """
    Один шар сцени.

    Шар — це GroupNode з додатковими властивостями:
    - Порядок рендерингу (render_order)
    - Заблокований від редагування (locked)
    - Теги для фільтрації
    """
    id:           str
    name:         str
    visible:      bool  = True
    opacity:      float = 1.0
    locked:       bool  = False
    render_order: int   = 0       # менше = рендерується першим
    tags:         dict[str, str] = field(default_factory=dict)

    # Вузли у цьому шарі
    _nodes: list[SceneNode] = field(default_factory=list, repr=False)

    def add(self, node: SceneNode) -> "Layer":
        """Додати вузол у шар."""
        if node not in self._nodes:
            self._nodes.append(node)
        return self

    def remove(self, node: SceneNode) -> bool:
        """Видалити вузол із шару."""
        if node in self._nodes:
            self._nodes.remove(node)
            return True
        return False

    def clear(self) -> None:
        """Видалити всі вузли зі шару."""
        self._nodes.clear()

    @property
    def nodes(self) -> list[SceneNode]:
        return list(self._nodes)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def iter_visible(self) -> Iterator[SceneNode]:
        """Ітерувати видимі вузли шару."""
        if not self.visible:
            return
        for node in self._nodes:
            if node.visible:
                yield node

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "name":         self.name,
            "visible":      self.visible,
            "opacity":      self.opacity,
            "locked":       self.locked,
            "render_order": self.render_order,
            "node_count":   self.node_count,
            "tags":         self.tags,
        }

    def __repr__(self) -> str:
        return (
            f"Layer(id={self.id!r}, name={self.name!r}, "
            f"nodes={self.node_count}, visible={self.visible})"
        )


# ----------------------------------------------------------------
# LAYER MANAGER
# ----------------------------------------------------------------

class LayerManager:
    """
    Менеджер шарів сцени.

    Підтримує:
    - Додавання / видалення шарів
    - Впорядкування (render order)
    - Пошук шарів за id/name
    - Видимість всіх шарів

    Usage:
        layers = LayerManager()
        layers.add(Layer("terrain",  "Base Terrain",  render_order=0))
        layers.add(Layer("satellite","Satellite",     render_order=1))
        layers.add(Layer("buildings","Buildings",     render_order=2))

        layers["buildings"].visible = False
        for layer in layers.ordered():
            render(layer)
    """

    def __init__(self) -> None:
        self._layers: dict[str, Layer] = {}

    # ---- Стандартні шари ----

    @classmethod
    def with_defaults(cls) -> "LayerManager":
        """Створити менеджер зі стандартними шарами GeoEngine."""
        mgr = cls()
        defaults = [
            Layer("base_terrain",   "Base Terrain",         render_order=0),
            Layer("satellite",      "Satellite Imagery",    render_order=1),
            Layer("hillshade",      "Hillshade",            render_order=2,
                  visible=False, opacity=0.5),
            Layer("osm_water",      "Water Bodies",         render_order=3),
            Layer("osm_roads",      "Roads",                render_order=4),
            Layer("osm_buildings",  "Buildings",            render_order=5),
            Layer("analysis",       "Analysis Overlay",     render_order=6,
                  visible=False, opacity=0.7),
            Layer("annotations",    "Annotations",          render_order=7),
        ]
        for layer in defaults:
            mgr.add(layer)
        return mgr

    # ---- CRUD ----

    def add(self, layer: Layer) -> "LayerManager":
        """Додати шар."""
        self._layers[layer.id] = layer
        return self

    def remove(self, layer_id: str) -> bool:
        """Видалити шар за ID."""
        if layer_id in self._layers:
            del self._layers[layer_id]
            return True
        return False

    def get(self, layer_id: str) -> Layer | None:
        """Отримати шар за ID."""
        return self._layers.get(layer_id)

    def __getitem__(self, layer_id: str) -> Layer:
        """Синтаксичний цукор: layers["buildings"]."""
        layer = self._layers.get(layer_id)
        if layer is None:
            raise KeyError(f"Шар {layer_id!r} не знайдено")
        return layer

    def __contains__(self, layer_id: str) -> bool:
        return layer_id in self._layers

    def find_by_name(self, name: str) -> Layer | None:
        """Пошук за ім'ям (перший збіг)."""
        for layer in self._layers.values():
            if layer.name == name:
                return layer
        return None

    # ---- Впорядкування та ітерація ----

    def ordered(self) -> list[Layer]:
        """Шари впорядковані за render_order (зростання)."""
        return sorted(self._layers.values(), key=lambda l: l.render_order)

    def visible_ordered(self) -> list[Layer]:
        """Тільки видимі шари впорядковані за render_order."""
        return [l for l in self.ordered() if l.visible]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.ordered())

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    # ---- Видимість ----

    def set_all_visible(self, visible: bool) -> None:
        for layer in self._layers.values():
            layer.visible = visible

    def show_only(self, *layer_ids: str) -> None:
        """Показати тільки вказані шари, решту сховати."""
        visible_set = set(layer_ids)
        for lid, layer in self._layers.items():
            layer.visible = lid in visible_set

    # ---- Серіалізація ----

    def to_dict(self) -> dict:
        return {
            "layers": [l.to_dict() for l in self.ordered()]
        }

    def __repr__(self) -> str:
        return f"LayerManager({self.layer_count} layers)"
