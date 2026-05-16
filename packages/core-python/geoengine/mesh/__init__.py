"""
GeoEngine — mesh пакет
3D Mesh генерація з DEM даних.
"""

from .terrain import (
    TerrainMesh,
    TerrainMeshBuilder,
)
from .lod import (
    LODConfig,
    LODManager,
    LODTileGrid,
    DEFAULT_LOD_PYRAMID,
)
from .normals import (
    generate_normal_map,
    generate_normal_map_16bit,
    save_normal_map,
)

__all__ = [
    # Terrain mesh
    "TerrainMesh",
    "TerrainMeshBuilder",
    # LOD
    "LODConfig",
    "LODManager",
    "LODTileGrid",
    "DEFAULT_LOD_PYRAMID",
    # Normal maps
    "generate_normal_map",
    "generate_normal_map_16bit",
    "save_normal_map",
]
