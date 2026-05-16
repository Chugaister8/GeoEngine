/**
 * GeoEngine — Quadtree LOD
 * Адаптивне розбиття простору для LOD системи терейну.
 *
 * Алгоритм:
 *   - Починаємо з одного кореневого вузла (весь світ або регіон)
 *   - Кожен вузол ділиться на 4 дочірні якщо:
 *     (a) камера близько до цього вузла
 *     (b) вузол ще не на максимальному LOD рівні
 *   - Leaf nodes → рендеримо
 *   - Frustum culling: відкидаємо невидимі вузли
 */

import * as THREE from 'three'
import type { BBox, TileXYZ, LODLevel } from '../../shared-types/src/geo'
import {
  tileToLatLonBBox,
  latLonToTile,
  haversineDistance,
  tileResolutionM,
} from '../geo/Coords'

// ----------------------------------------------------------------
// КОНФІГУРАЦІЯ LOD
// ----------------------------------------------------------------

export interface LODConfig {
  readonly level:          LODLevel
  readonly maxDistanceM:   number    // максимальна відстань камери
  readonly targetResM:     number    // цільова роздільна здатність
}

export const DEFAULT_LOD_CONFIGS: LODConfig[] = [
  { level: 0, maxDistanceM:   2_000, targetResM:    1 },
  { level: 1, maxDistanceM:  10_000, targetResM:    5 },
  { level: 2, maxDistanceM:  50_000, targetResM:   25 },
  { level: 3, maxDistanceM: 200_000, targetResM:  100 },
  { level: 4, maxDistanceM: 500_000, targetResM:  500 },
  { level: 5, maxDistanceM: Infinity, targetResM: 2500 },
]

// Zoom рівні відповідні LOD рівням
const LOD_TO_ZOOM: Record<LODLevel, number> = {
  0: 14,   // ~1-5m/px
  1: 12,   // ~10m/px
  2: 10,   // ~40m/px
  3: 8,    // ~150m/px
  4: 6,    // ~600m/px
  5: 4,    // ~2500m/px
}

// ----------------------------------------------------------------
// QUADTREE NODE
// ----------------------------------------------------------------

export type QuadtreeNodeState =
  | 'idle'      // не ініціалізований
  | 'active'    // leaf node — рендеримо
  | 'split'     // розбитий — не рендеримо, рендеруємо дочірні
  | 'culled'    // поза frustum — не рендеримо

export class QuadtreeNode {
  readonly tile:     TileXYZ
  readonly bbox:     BBox
  readonly level:    LODLevel
  readonly depth:    number

  state:    QuadtreeNodeState = 'idle'
  children: [QuadtreeNode, QuadtreeNode, QuadtreeNode, QuadtreeNode] | null = null

  // Відстань від камери до центру (оновлюється кожен кадр)
  distanceToCamera: number = Infinity

  constructor(tile: TileXYZ, depth = 0) {
    this.tile  = tile
    this.bbox  = tileToLatLonBBox(tile)
    this.level = Math.min(depth, 5) as LODLevel
    this.depth = depth
  }

  get isLeaf(): boolean {
    return this.children === null
  }

  get centerLat(): number {
    return (this.bbox.north + this.bbox.south) / 2
  }

  get centerLon(): number {
    return (this.bbox.west + this.bbox.east) / 2
  }

  get tileKey(): string {
    return `${this.tile.z}/${this.tile.x}/${this.tile.y}`
  }

  /**
   * Розбити вузол на 4 дочірніх.
   * Дочірні відповідають чотирьом квадрантам тайлу.
   */
  split(): void {
    if (!this.isLeaf) return
    const { x, y, z } = this.tile
    const childZ = z + 1
    this.children = [
      new QuadtreeNode({ x: x*2,   y: y*2,   z: childZ }, this.depth + 1), // TL
      new QuadtreeNode({ x: x*2+1, y: y*2,   z: childZ }, this.depth + 1), // TR
      new QuadtreeNode({ x: x*2,   y: y*2+1, z: childZ }, this.depth + 1), // BL
      new QuadtreeNode({ x: x*2+1, y: y*2+1, z: childZ }, this.depth + 1), // BR
    ]
    this.state = 'split'
  }

  /**
   * Злити дочірні вузли назад (merge).
   * Відбувається коли камера відлітає далеко.
   */
  merge(): void {
    if (this.isLeaf) return
    // Рекурсивно мерджимо дочірні
    for (const child of this.children!) {
      child.merge()
    }
    this.children = null
    this.state    = 'active'
  }

  /**
   * Зібрати всі leaf nodes рекурсивно.
   */
  collectLeaves(out: QuadtreeNode[] = []): QuadtreeNode[] {
    if (this.state === 'culled') return out
    if (this.isLeaf) {
      out.push(this)
      return out
    }
    for (const child of this.children!) {
      child.collectLeaves(out)
    }
    return out
  }
}

// ----------------------------------------------------------------
// QUADTREE LOD MANAGER
// ----------------------------------------------------------------

export class QuadtreeLOD {
  /**
   * Головний LOD менеджер на основі Quadtree.
   *
   * Алгоритм оновлення (викликається кожен кадр):
   * 1. Оновити відстань до камери для кожного вузла
   * 2. Frustum cull: відмітити невидимі вузли як 'culled'
   * 3. Traverse quadtree:
   *    - Якщо вузол занадто далеко → merge (або не split)
   *    - Якщо вузол близько → split (якщо не на max LOD)
   * 4. Зібрати leaf nodes → це список тайлів для рендерингу
   *
   * Usage:
   *   const lod = new QuadtreeLOD({ rootZoom: 4 })
   *   // кожен кадр:
   *   const tiles = lod.update(camera, frustum)
   *   // tiles — список TileXYZ для завантаження/рендерингу
   */

  private roots:   QuadtreeNode[]
  private configs: LODConfig[]
  private maxZoom: number

  constructor(options: {
    rootZoom?:  number
    maxZoom?:   number
    configs?:   LODConfig[]
    rootBBox?:  BBox    // обмежений регіон, або весь світ
  } = {}) {
    const {
      rootZoom = 4,
      maxZoom  = 14,
      configs  = DEFAULT_LOD_CONFIGS,
      rootBBox,
    } = options

    this.configs = configs
    this.maxZoom = maxZoom
    this.roots   = this._buildRoots(rootZoom, rootBBox)
  }

  /**
   * Оновити quadtree за позицією камери та frustum.
   * Викликати кожен кадр (або при зміні позиції камери).
   *
   * @param cameraLat  широта камери
   * @param cameraLon  довгота камери
   * @param cameraAlt  висота камери (метри)
   * @param frustum    Three.js Frustum для culling
   *
   * @returns список leaf nodes (тайлів) для рендерингу
   */
  update(
    cameraLat: number,
    cameraLon: number,
    cameraAlt: number,
    frustum:   THREE.Frustum,
  ): QuadtreeNode[] {
    // Оновлюємо кожен корінь
    for (const root of this.roots) {
      this._updateNode(root, cameraLat, cameraLon, cameraAlt, frustum)
    }

    // Збираємо всі видимі leaf nodes
    const leaves: QuadtreeNode[] = []
    for (const root of this.roots) {
      root.collectLeaves(leaves)
    }

    return leaves
  }

  /**
   * Отримати список TileXYZ для рендерингу (без дублікатів).
   */
  getVisibleTiles(
    cameraLat: number,
    cameraLon: number,
    cameraAlt: number,
    frustum:   THREE.Frustum,
  ): Array<{ tile: TileXYZ; lodLevel: LODLevel; distanceM: number }> {
    const leaves = this.update(cameraLat, cameraLon, cameraAlt, frustum)

    return leaves
      .filter(n => n.state !== 'culled')
      .map(n => ({
        tile:      n.tile,
        lodLevel:  n.level,
        distanceM: n.distanceToCamera,
      }))
      .sort((a, b) => a.distanceM - b.distanceM)  // ближні першими
  }

  // ---- Статистика ----

  getStats(): QuadtreeStats {
    let total     = 0
    let leaves    = 0
    let culled    = 0
    let maxDepth  = 0

    const traverse = (node: QuadtreeNode, depth: number): void => {
      total++
      maxDepth = Math.max(maxDepth, depth)
      if (node.state === 'culled') { culled++; return }
      if (node.isLeaf) { leaves++; return }
      for (const child of node.children!) {
        traverse(child, depth + 1)
      }
    }

    for (const root of this.roots) traverse(root, 0)

    return { total, leaves, culled, maxDepth }
  }

  // ---- Приватні методи ----

  private _updateNode(
    node:      QuadtreeNode,
    camLat:    number,
    camLon:    number,
    camAlt:    number,
    frustum:   THREE.Frustum,
  ): void {
    // 1. Frustum culling
    if (!this._nodeInFrustum(node, frustum)) {
      node.state = 'culled'
      return
    }

    // 2. Відстань від камери до центру тайлу
    const groundDist = haversineDistance(
      camLat, camLon,
      node.centerLat, node.centerLon,
    )
    // Враховуємо висоту камери (3D відстань приблизно)
    node.distanceToCamera = Math.sqrt(groundDist ** 2 + camAlt ** 2)

    // 3. Визначити потрібний LOD рівень
    const targetLOD = this._selectLOD(node.distanceToCamera)

    // 4. Split або merge
    const atMaxZoom = node.tile.z >= this.maxZoom

    if (targetLOD < node.depth && !atMaxZoom) {
      // Потрібна більша деталізація → split
      if (node.isLeaf) {
        node.split()
      }
      // Рекурсивно оновлюємо дочірні
      for (const child of node.children!) {
        this._updateNode(child, camLat, camLon, camAlt, frustum)
      }
    } else {
      // Достатня деталізація або max zoom → merge і стати leaf
      if (!node.isLeaf) {
        node.merge()
      }
      node.state = 'active'
    }
  }

  private _selectLOD(distanceM: number): number {
    for (const config of this.configs) {
      if (distanceM <= config.maxDistanceM) {
        return config.level
      }
    }
    return this.configs[this.configs.length - 1]!.level
  }

  private _nodeInFrustum(node: QuadtreeNode, frustum: THREE.Frustum): boolean {
    // Спрощений AABB frustum culling
    // Використовуємо центр + половину діагоналі як bounding sphere
    const bbox = node.bbox
    const centerLat = (bbox.north + bbox.south) / 2
    const centerLon = (bbox.west  + bbox.east)  / 2
    const halfDiag  = haversineDistance(
      centerLat, centerLon,
      bbox.north, bbox.east,
    )

    // Three.js Sphere для frustum перевірки
    const sphere = new THREE.Sphere(
      new THREE.Vector3(centerLon, 0, -centerLat),
      halfDiag / 111_320,  // конвертуємо метри → умовні одиниці
    )

    return frustum.intersectsSphere(sphere)
  }

  private _buildRoots(rootZoom: number, bbox?: BBox): QuadtreeNode[] {
    // Якщо bbox заданий — тільки ті тайли що перетинаються
    const worldBBox = bbox ?? {
      west: -180, south: -85.051129,
      east:  180, north:  85.051129,
    }

    const n = 1 << rootZoom  // 2^zoom
    const roots: QuadtreeNode[] = []

    const tl = latLonToTile(worldBBox.north, worldBBox.west, rootZoom)
    const br = latLonToTile(worldBBox.south, worldBBox.east, rootZoom)

    for (let y = tl.y; y <= br.y; y++) {
      for (let x = tl.x; x <= br.x; x++) {
        roots.push(new QuadtreeNode({ x, y, z: rootZoom }, 0))
      }
    }

    return roots
  }
}

// ----------------------------------------------------------------
// ТИПИ СТАТИСТИКИ
// ----------------------------------------------------------------

export interface QuadtreeStats {
  total:    number
  leaves:   number
  culled:   number
  maxDepth: number
  }
