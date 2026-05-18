"""GeoEngine — simulation пакет."""

from .ballistics import (
    BallisticsSolver,
    BallisticsResult,
    ProjectileParams,
    WindVector,
    LatLonAlt,
    BALLISTIC_PRESETS,
    optimal_elevation,
)
from .fire import (
    FireSimulation,
    FireResult,
    FireCell,
    CellState,
    FUEL_TYPES,
)

__all__ = [
    "BallisticsSolver", "BallisticsResult",
    "ProjectileParams", "WindVector", "LatLonAlt",
    "BALLISTIC_PRESETS", "optimal_elevation",
    "FireSimulation", "FireResult",
    "FireCell", "CellState", "FUEL_TYPES",
]
