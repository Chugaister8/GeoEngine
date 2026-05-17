"""
GeoEngine — glTF 2.0 Export
Експорт TerrainMesh та BuildingMesh у glTF 2.0 / GLB формат.

glTF = GL Transmission Format — стандарт 3D моделей для WebGPU/Three.js.
GLB  = бінарна версія glTF (один файл).

Spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

Структура glTF:
  asset:        метадані
  scene/scenes: список сцен
  nodes:        вузли
  meshes:       геометрія
  accessors:    типізований доступ до буферів
  bufferViews:  зрізи буфера
  buffers:      бінарні дані (base64 або .bin)
  materials:    матеріали (PBR)
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from ..mesh.terrain   import TerrainMesh
from ..osm.buildings  import BuildingMesh, BuildingCollection

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# GLTF CONSTANTS
# ----------------------------------------------------------------

GLTF_FLOAT       = 5126   # GL_FLOAT
GLTF_UNSIGNED_INT = 5125  # GL_UNSIGNED_INT
GLTF_ARRAY_BUFFER = 34962
GLTF_ELEMENT_ARRAY_BUFFER = 34963

GLTF_VEC2 = "VEC2"
GLTF_VEC3 = "VEC3"
GLTF_SCALAR = "SCALAR"


# ----------------------------------------------------------------
# GLTF BUILDER
# ----------------------------------------------------------------

class GLTFBuilder:
    """
    Будівник glTF 2.0 документів.

    Usage:
        builder = GLTFBuilder()
        builder.add_terrain_mesh(terrain_mesh, name="Carpathians")
        gltf_dict = builder.build()
        builder.save_glb("output.glb")
    """

    def __init__(self) -> None:
        self._gltf:        dict[str, Any] = {
            "asset":   {"version": "2.0", "generator": "GeoEngine 0.1.0"},
            "scene":   0,
            "scenes":  [{"nodes": []}],
            "nodes":   [],
            "meshes":  [],
            "accessors": [],
            "bufferViews": [],
            "buffers":  [],
            "materials": [],
        }
        self._bin_data:    list[bytes] = []
        self._bin_offset:  int         = 0

    # ---- Terrain ----

    def add_terrain_mesh(
        self,
        mesh: TerrainMesh,
        name: str = "terrain",
    ) -> int:
        """
        Додати TerrainMesh до glTF.

        Returns:
            Індекс вузла у gltf["nodes"]
        """
        mesh_idx = self._add_mesh_buffers(
            name=name,
            vertices=mesh.vertices,
            indices=mesh.indices,
            uvs=mesh.uvs,
            normals=mesh.normals,
        )
        node_idx = self._add_node(name=name, mesh=mesh_idx)
        self._gltf["scenes"][0]["nodes"].append(node_idx)
        return node_idx

    def add_building_collection(
        self,
        collection: BuildingCollection,
        name: str = "buildings",
    ) -> int:
        """
        Додати BuildingCollection до glTF (один node per building).

        Returns:
            Індекс group node
        """
        child_nodes: list[int] = []

        for bm in collection.meshes:
            bname    = f"{bm.building_type}_{bm.osm_id}"
            mat_idx  = self._add_pbr_material(
                name=bname,
                color=(*bm.color, 1.0),
            )
            mesh_idx = self._add_mesh_buffers(
                name=bname,
                vertices=bm.vertices,
                indices=bm.indices,
                uvs=bm.uvs,
                normals=bm.normals,
                material=mat_idx,
            )
            node_idx = self._add_node(name=bname, mesh=mesh_idx)
            child_nodes.append(node_idx)

        group_idx = len(self._gltf["nodes"])
        self._gltf["nodes"].append({
            "name":     name,
            "children": child_nodes,
        })
        self._gltf["scenes"][0]["nodes"].append(group_idx)

        log.info(
            "gltf.buildings",
            count=len(collection.meshes),
            group_node=group_idx,
        )
        return group_idx

    # ---- Export ----

    def build(self) -> dict:
        """Повернути готовий glTF словник."""
        # Об'єднати всі бінарні дані в один buffer
        combined  = b"".join(self._bin_data)
        encoded   = base64.b64encode(combined).decode("ascii")
        data_uri  = f"data:application/octet-stream;base64,{encoded}"

        self._gltf["buffers"] = [{
            "byteLength": len(combined),
            "uri":        data_uri,
        }]
        return self._gltf

    def save_gltf(self, path: str | Path) -> Path:
        """Зберегти як .gltf (JSON з embedded base64 даними)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.build(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("io.gltf.save", path=str(path))
        return path

    def save_glb(self, path: str | Path) -> Path:
        """
        Зберегти як .glb (бінарний glTF).

        GLB структура:
          [12 byte header][JSON chunk][BIN chunk]
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Підготувати дані
        bin_data = b"".join(self._bin_data)

        # Клонуємо gltf без URI (для GLB)
        gltf_copy = json.loads(json.dumps(self._gltf))
        gltf_copy["buffers"] = [{"byteLength": len(bin_data)}]
        json_str  = json.dumps(gltf_copy, separators=(",", ":"))

        # JSON chunk (вирівнювання до 4 байт пробілами)
        json_bytes = json_str.encode("utf-8")
        json_pad   = (4 - len(json_bytes) % 4) % 4
        json_chunk = json_bytes + b" " * json_pad

        # BIN chunk (вирівнювання до 4 байт нулями)
        bin_pad    = (4 - len(bin_data) % 4) % 4
        bin_chunk  = bin_data + b"\x00" * bin_pad

        # Header
        magic      = b"glTF"
        version    = struct.pack("<I", 2)
        total_len  = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        length     = struct.pack("<I", total_len)

        with open(path, "wb") as f:
            # Header
            f.write(magic + version + length)
            # JSON chunk
            f.write(struct.pack("<I", len(json_chunk)))  # chunk length
            f.write(struct.pack("<I", 0x4E4F534A))       # "JSON"
            f.write(json_chunk)
            # BIN chunk
            f.write(struct.pack("<I", len(bin_chunk)))   # chunk length
            f.write(struct.pack("<I", 0x004E4942))       # "BIN\x00"
            f.write(bin_chunk)

        log.info("io.gltf.save_glb", path=str(path), size_mb=total_len/1e6)
        return path

    # ---- Приватні методи ----

    def _add_buffer_view(
        self,
        data:   bytes,
        target: int,
    ) -> int:
        """Додати bufferView та повернути його індекс."""
        idx = len(self._gltf["bufferViews"])
        self._gltf["bufferViews"].append({
            "buffer":     0,
            "byteOffset": self._bin_offset,
            "byteLength": len(data),
            "target":     target,
        })
        self._bin_data.append(data)
        self._bin_offset += len(data)
        # Вирівнювання до 4 байт
        pad = (4 - self._bin_offset % 4) % 4
        if pad:
            self._bin_data.append(b"\x00" * pad)
            self._bin_offset += pad
        return idx

    def _add_accessor(
        self,
        buffer_view: int,
        count:       int,
        component_type: int,
        accessor_type:  str,
        min_vals:    list[float] | None = None,
        max_vals:    list[float] | None = None,
    ) -> int:
        """Додати accessor та повернути його індекс."""
        idx = len(self._gltf["accessors"])
        acc: dict[str, Any] = {
            "bufferView":    buffer_view,
            "componentType": component_type,
            "count":         count,
            "type":          accessor_type,
        }
        if min_vals is not None:
            acc["min"] = min_vals
        if max_vals is not None:
            acc["max"] = max_vals
        self._gltf["accessors"].append(acc)
        return idx

    def _add_mesh_buffers(
        self,
        name:      str,
        vertices:  npt.NDArray[np.float32],
        indices:   npt.NDArray[np.uint32],
        uvs:       npt.NDArray[np.float32],
        normals:   npt.NDArray[np.float32],
        material:  int | None = None,
    ) -> int:
        """Додати буфери меша та повернути індекс mesh."""
        # Positions
        pos_bv  = self._add_buffer_view(vertices.tobytes(), GLTF_ARRAY_BUFFER)
        pos_min = vertices.min(axis=0).tolist()
        pos_max = vertices.max(axis=0).tolist()
        pos_acc = self._add_accessor(
            pos_bv, len(vertices), GLTF_FLOAT, GLTF_VEC3,
            min_vals=pos_min, max_vals=pos_max,
        )

        # Indices
        idx_data = indices.flatten().astype(np.uint32)
        idx_bv   = self._add_buffer_view(idx_data.tobytes(), GLTF_ELEMENT_ARRAY_BUFFER)
        idx_acc  = self._add_accessor(
            idx_bv, len(idx_data), GLTF_UNSIGNED_INT, GLTF_SCALAR,
        )

        # UVs
        uv_bv  = self._add_buffer_view(uvs.tobytes(), GLTF_ARRAY_BUFFER)
        uv_acc = self._add_accessor(uv_bv, len(uvs), GLTF_FLOAT, GLTF_VEC2)

        # Normals
        nm_bv  = self._add_buffer_view(normals.tobytes(), GLTF_ARRAY_BUFFER)
        nm_acc = self._add_accessor(nm_bv, len(normals), GLTF_FLOAT, GLTF_VEC3)

        # Primitive
        primitive: dict[str, Any] = {
            "attributes": {
                "POSITION": pos_acc,
                "TEXCOORD_0": uv_acc,
                "NORMAL": nm_acc,
            },
            "indices": idx_acc,
            "mode": 4,   # TRIANGLES
        }
        if material is not None:
            primitive["material"] = material

        # Mesh
        mesh_idx = len(self._gltf["meshes"])
        self._gltf["meshes"].append({
            "name":       name,
            "primitives": [primitive],
        })
        return mesh_idx

    def _add_node(self, name: str, mesh: int) -> int:
        """Додати node та повернути його індекс."""
        idx = len(self._gltf["nodes"])
        self._gltf["nodes"].append({"name": name, "mesh": mesh})
        return idx

    def _add_pbr_material(
        self,
        name:  str,
        color: tuple[float, float, float, float] = (0.8, 0.7, 0.6, 1.0),
        roughness: float = 0.9,
        metallic:  float = 0.0,
    ) -> int:
        """Додати PBR матеріал."""
        idx = len(self._gltf["materials"])
        self._gltf["materials"].append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor":          list(color),
                "metallicFactor":           metallic,
                "roughnessFactor":          roughness,
            },
            "doubleSided": False,
        })
        return idx


# ----------------------------------------------------------------
# ЗРУЧНІ ФУНКЦІЇ
# ----------------------------------------------------------------

def terrain_to_gltf(
    mesh:     TerrainMesh,
    path:     str | Path,
    format:   str = "glb",
    name:     str = "terrain",
) -> Path:
    """
    Зберегти TerrainMesh як glTF/GLB.

    Args:
        mesh:   TerrainMesh
        path:   вихідний файл (.gltf або .glb)
        format: "gltf" або "glb"
        name:   ім'я mesh у файлі

    Returns:
        Path до збереженого файлу
    """
    builder = GLTFBuilder()
    builder.add_terrain_mesh(mesh, name=name)

    if format == "glb":
        return builder.save_glb(path)
    return builder.save_gltf(path)


def buildings_to_gltf(
    collection: BuildingCollection,
    path:       str | Path,
    format:     str = "glb",
) -> Path:
    """Зберегти BuildingCollection як glTF/GLB."""
    builder = GLTFBuilder()
    builder.add_building_collection(collection, name="buildings")

    if format == "glb":
        return builder.save_glb(path)
    return builder.save_gltf(path)
