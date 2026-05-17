"""
GeoEngine — Terrain Mesh Builder Tests
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from geoengine.mesh.terrain import (
    TerrainMesh,
    TerrainMeshBuilder,
    _build_grid_indices,
    _compute_normals,
)
from geoengine.mesh.lod import (
    LODConfig,
    LODManager,
    DEFAULT_LOD_PYRAMID,
)
from geoengine.mesh.normals import generate_normal_map
from geoengine.geo.coords import LLH


# ================================================================
# GRID INDICES
# ================================================================

class TestGridIndices:

    def test_2x2_grid(self):
        """2×2 → 2 трикутники."""
        idx = _build_grid_indices(w=2, h=2)
        assert idx.shape == (2, 3)

    def test_3x3_grid(self):
        """3×3 → (3-1)×(3-1)×2 = 8 трикутників."""
        idx = _build_grid_indices(w=3, h=3)
        assert idx.shape == (8, 3)

    def test_all_indices_valid(self):
        """Всі індекси в діапазоні [0, w*h)."""
        w, h = 5, 4
        idx  = _build_grid_indices(w=w, h=h)
        assert np.all(idx >= 0)
        assert np.all(idx < w * h)

    def test_no_degenerate_triangles(self):
        """Немає вироджених трикутників (всі 3 індекси різні)."""
        idx = _build_grid_indices(w=4, h=4)
        for tri in idx:
            assert len(set(tri)) == 3

    def test_dtype_uint32(self):
        idx = _build_grid_indices(w=3, h=3)
        assert idx.dtype == np.uint32


# ================================================================
# NORMALS
# ================================================================

class TestNormals:

    def test_flat_surface_normal_up(self):
        """Плоска горизонтальна поверхня → нормаль вгору (0,1,0)."""
        # Плоска сітка 2×2
        vertices = np.array([
            [0, 0, 0], [1, 0, 0],
            [0, 0, 1], [1, 0, 1],
        ], dtype=np.float32)
        indices = np.array([[0, 2, 1], [1, 2, 3]], dtype=np.uint32)

        normals = _compute_normals(vertices, indices)

        # Всі нормалі мають вказувати вгору (Y=1)
        for n in normals:
            assert abs(n[1]) > 0.9, f"Normal should point up: {n}"

    def test_normals_normalized(self, synthetic_dem_tile):
        """Нормалі мають одиничну довжину."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=256)

        lengths = np.linalg.norm(mesh.normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-5)

    def test_normals_shape(self, synthetic_dem_tile):
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=256)

        assert mesh.normals.shape == (mesh.vertex_count, 3)


# ================================================================
# TERRAIN MESH BUILDER
# ================================================================

class TestTerrainMeshBuilder:

    def test_basic_build(self, synthetic_dem_tile):
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile)

        assert isinstance(mesh, TerrainMesh)
        assert mesh.vertex_count   > 0
        assert mesh.triangle_count > 0

    def test_vertices_shape(self, synthetic_dem_tile):
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile)

        assert mesh.vertices.ndim  == 2
        assert mesh.vertices.shape[1] == 3
        assert mesh.indices.shape[1]  == 3
        assert mesh.uvs.shape[1]      == 2

    def test_uv_in_range(self, synthetic_dem_tile):
        """UV координати в [0..1]."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=1024)

        assert np.all(mesh.uvs >= 0.0)
        assert np.all(mesh.uvs <= 1.0)

    def test_max_verts_limit(self, synthetic_dem_tile):
        """max_verts обмежує кількість вершин."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=256)

        # Без skirt: vertex_count ≤ max_verts
        assert mesh.vertex_count <= 256 + 100   # +100 допуск для округлення

    def test_skirt_increases_vertex_count(self, synthetic_dem_tile):
        """Skirt додає вершини."""
        origin = LLH(lat=48.0, lon=23.0)
        b_no_skirt = TerrainMeshBuilder(origin=origin, skirt_height=0.0)
        b_skirt    = TerrainMeshBuilder(origin=origin, skirt_height=100.0)

        m_no_skirt = b_no_skirt.build(synthetic_dem_tile, max_verts=512)
        m_skirt    = b_skirt.build(synthetic_dem_tile,    max_verts=512)

        assert m_skirt.vertex_count > m_no_skirt.vertex_count

    def test_adaptive_build(self, synthetic_dem_tile):
        """Adaptive метод будує валідний меш."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(
            synthetic_dem_tile,
            method="adaptive",
            max_verts=2048,
        )

        assert mesh.vertex_count   > 0
        assert mesh.triangle_count > 0
        # Adaptive має менше вершин ніж uniform (для плоских ділянок)

    def test_y_axis_is_elevation(self, synthetic_dem_tile):
        """Y координата вершин відповідає висоті (Three.js convention)."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=512)

        # Y (index 1) — висота над ENU origin
        y_values = mesh.vertices[:, 1]
        assert y_values.max() >= 0   # принаймні якась висота > 0

    def test_serialization(self, synthetic_dem_tile):
        """to_dict() повертає коректний словник."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=256)

        d = mesh.to_dict()
        assert d["type"]           == "terrain_mesh"
        assert d["vertex_count"]   == mesh.vertex_count
        assert d["triangle_count"] == mesh.triangle_count
        assert "vertices" in d["buffers"]
        assert "indices"  in d["buffers"]
        assert "uvs"      in d["buffers"]
        assert "normals"  in d["buffers"]

    def test_memory_mb_reasonable(self, synthetic_dem_tile):
        """Пам'ять меша в розумних межах."""
        origin  = LLH(lat=48.0, lon=23.0)
        builder = TerrainMeshBuilder(origin=origin, skirt_height=0)
        mesh    = builder.build(synthetic_dem_tile, max_verts=1024)

        # Для 1024 вершин: 1024 × (3+2+3) float32 × 4B + indices
        # ≈ 30-60KB
        assert mesh.memory_mb < 10.0   # < 10MB (дуже консервативно)
        assert mesh.memory_mb > 0.001  # > 1KB


# ================================================================
# LOD MANAGER
# ================================================================

class TestLODManager:

    def test_get_mesh_for_level(self, synthetic_dem_tile):
        origin  = LLH(lat=48.0, lon=23.0)
        manager = LODManager(tile=synthetic_dem_tile, origin=origin)

        mesh = manager.get_mesh_for_level(0)
        assert isinstance(mesh, TerrainMesh)

    def test_higher_lod_has_fewer_verts(self, synthetic_dem_tile):
        """LOD 2 має менше вершин ніж LOD 0."""
        origin  = LLH(lat=48.0, lon=23.0)
        manager = LODManager(tile=synthetic_dem_tile, origin=origin)

        mesh0 = manager.get_mesh_for_level(0)
        mesh2 = manager.get_mesh_for_level(2)

        assert mesh2.vertex_count <= mesh0.vertex_count

    def test_caching(self, synthetic_dem_tile):
        """Повторний запит повертає той самий об'єкт (кеш)."""
        origin  = LLH(lat=48.0, lon=23.0)
        manager = LODManager(tile=synthetic_dem_tile, origin=origin)

        mesh_a = manager.get_mesh_for_level(0)
        mesh_b = manager.get_mesh_for_level(0)
        assert mesh_a is mesh_b   # той самий об'єкт

    def test_get_mesh_for_distance(self, synthetic_dem_tile):
        """Близька відстань → детальніший LOD."""
        origin  = LLH(lat=48.0, lon=23.0)
        manager = LODManager(tile=synthetic_dem_tile, origin=origin)

        mesh_close = manager.get_mesh_for_distance(500)
        mesh_far   = manager.get_mesh_for_distance(200_000)

        assert mesh_close.vertex_count >= mesh_far.vertex_count

    def test_memory_usage(self, synthetic_dem_tile):
        origin  = LLH(lat=48.0, lon=23.0)
        manager = LODManager(tile=synthetic_dem_tile, origin=origin)

        manager.get_mesh_for_level(0)
        manager.get_mesh_for_level(1)

        assert manager.memory_usage_mb > 0

    def test_clear_cache(self, synthetic_dem_tile):
        origin  = LLH(lat=48.0, lon=23.0)
        manager = LODManager(tile=synthetic_dem_tile, origin=origin)

        manager.get_mesh_for_level(0)
        manager.clear_cache()
        assert manager.memory_usage_mb == 0.0


# ================================================================
# NORMAL MAP GENERATION
# ================================================================

class TestNormalMapGeneration:

    def test_normal_map_shape(self, synthetic_dem_tile):
        nm = generate_normal_map(synthetic_dem_tile)
        h, w = synthetic_dem_tile.data.shape
        assert nm.shape == (h, w, 3)

    def test_normal_map_dtype(self, synthetic_dem_tile):
        nm = generate_normal_map(synthetic_dem_tile)
        assert nm.dtype == np.uint8

    def test_normal_map_range(self, synthetic_dem_tile):
        nm = generate_normal_map(synthetic_dem_tile)
        assert nm.min() >= 0
        assert nm.max() <= 255

    def test_flat_terrain_neutral_normal(self, bbox_small):
        """Плоский терейн → нейтральна нормаль (128, 255, 128)."""
        from rasterio.transform import from_bounds
        from geoengine.dem.loader import DEMTile

        flat  = np.full((32, 32), 500.0, dtype=np.float32)
        t     = from_bounds(bbox_small.west, bbox_small.south, bbox_small.east, bbox_small.north, 32, 32)
        tile  = DEMTile(data=flat, bbox=bbox_small, transform=t, crs="EPSG:4326", source="flat")

        nm = generate_normal_map(tile)

        # Внутрішні пікселі мають G ≈ 255 (нормаль вгору)
        inner_g = nm[2:-2, 2:-2, 1]
        assert np.mean(inner_g) > 200   # переважно 255

    def test_normal_map_strength(self, synthetic_dem_tile):
        """strength > 1 дає більший контраст."""
        nm1 = generate_normal_map(synthetic_dem_tile, strength=1.0)
        nm2 = generate_normal_map(synthetic_dem_tile, strength=3.0)

        # G канал (up) менший при більшій силі (більший нахил)
        assert nm2[:, :, 1].mean() < nm1[:, :, 1].mean()
