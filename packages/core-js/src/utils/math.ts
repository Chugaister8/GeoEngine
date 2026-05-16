/**
 * GeoEngine — Math Utilities (JavaScript)
 * Дзеркало Python math3d.py для браузера.
 * Tree-shakeable, zero-dependency.
 */

// ----------------------------------------------------------------
// CONSTANTS
// ----------------------------------------------------------------

export const PI     = Math.PI
export const TWO_PI = Math.PI * 2
export const HALF_PI = Math.PI * 0.5
export const DEG2RAD = Math.PI / 180
export const RAD2DEG = 180 / Math.PI
export const EPSILON = 1e-7

// ----------------------------------------------------------------
// SCALAR UTILS
// ----------------------------------------------------------------

export const lerp = (a: number, b: number, t: number): number =>
  a + (b - a) * t

export const clamp = (v: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, v))

export const clamp01 = (v: number): number =>
  Math.max(0, Math.min(1, v))

export const smoothstep = (edge0: number, edge1: number, x: number): number => {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

export const smootherstep = (edge0: number, edge1: number, x: number): number => {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * t * (t * (t * 6 - 15) + 10)
}

export const easeInOutCubic = (t: number): number =>
  t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2

export const easeOutQuart = (t: number): number =>
  1 - (1 - t) ** 4

export const degToRad = (deg: number): number => deg * DEG2RAD
export const radToDeg = (rad: number): number => rad * RAD2DEG

export const angleWrap360 = (deg: number): number => ((deg % 360) + 360) % 360
export const angleWrap180 = (deg: number): number => {
  const a = angleWrap360(deg)
  return a > 180 ? a - 360 : a
}

// ----------------------------------------------------------------
// VEC2
// ----------------------------------------------------------------

export type Vec2 = readonly [number, number]

export const vec2 = (x: number, y: number): Vec2 => [x, y]
export const vec2Zero  = (): Vec2 => [0, 0]
export const vec2One   = (): Vec2 => [1, 1]

export const vec2Add   = ([ax,ay]: Vec2, [bx,by]: Vec2): Vec2 => [ax+bx, ay+by]
export const vec2Sub   = ([ax,ay]: Vec2, [bx,by]: Vec2): Vec2 => [ax-bx, ay-by]
export const vec2Scale = ([x,y]: Vec2, s: number):        Vec2 => [x*s,   y*s]
export const vec2Dot   = ([ax,ay]: Vec2, [bx,by]: Vec2): number => ax*bx + ay*by
export const vec2Len   = ([x,y]: Vec2): number => Math.sqrt(x*x + y*y)
export const vec2LenSq = ([x,y]: Vec2): number => x*x + y*y

export const vec2Norm  = (v: Vec2): Vec2 => {
  const l = vec2Len(v)
  return l < EPSILON ? [0,0] : [v[0]/l, v[1]/l]
}

export const vec2Lerp  = (a: Vec2, b: Vec2, t: number): Vec2 => [
  lerp(a[0], b[0], t),
  lerp(a[1], b[1], t),
]

// ----------------------------------------------------------------
// VEC3
// ----------------------------------------------------------------

export type Vec3 = readonly [number, number, number]

export const vec3 = (x: number, y: number, z: number): Vec3 => [x, y, z]
export const vec3Zero    = (): Vec3 => [0, 0, 0]
export const vec3One     = (): Vec3 => [1, 1, 1]
export const vec3Up      = (): Vec3 => [0, 1, 0]
export const vec3Forward = (): Vec3 => [0, 0, -1]
export const vec3Right   = (): Vec3 => [1, 0, 0]

export const vec3Add   = ([ax,ay,az]: Vec3, [bx,by,bz]: Vec3): Vec3 =>
  [ax+bx, ay+by, az+bz]
export const vec3Sub   = ([ax,ay,az]: Vec3, [bx,by,bz]: Vec3): Vec3 =>
  [ax-bx, ay-by, az-bz]
export const vec3Scale = ([x,y,z]: Vec3, s: number): Vec3 =>
  [x*s, y*s, z*s]
export const vec3Neg   = ([x,y,z]: Vec3): Vec3 => [-x,-y,-z]

export const vec3Dot   = ([ax,ay,az]: Vec3, [bx,by,bz]: Vec3): number =>
  ax*bx + ay*by + az*bz

export const vec3Cross = ([ax,ay,az]: Vec3, [bx,by,bz]: Vec3): Vec3 => [
  ay*bz - az*by,
  az*bx - ax*bz,
  ax*by - ay*bx,
]

export const vec3Len   = ([x,y,z]: Vec3): number => Math.sqrt(x*x + y*y + z*z)
export const vec3LenSq = ([x,y,z]: Vec3): number => x*x + y*y + z*z

export const vec3Norm  = (v: Vec3): Vec3 => {
  const l = vec3Len(v)
  return l < EPSILON ? [0,1,0] : [v[0]/l, v[1]/l, v[2]/l]
}

export const vec3Lerp  = (a: Vec3, b: Vec3, t: number): Vec3 => [
  lerp(a[0], b[0], t),
  lerp(a[1], b[1], t),
  lerp(a[2], b[2], t),
]

export const vec3Dist  = (a: Vec3, b: Vec3): number =>
  vec3Len(vec3Sub(b, a))

export const vec3DistSq = (a: Vec3, b: Vec3): number =>
  vec3LenSq(vec3Sub(b, a))

export const vec3Reflect = (v: Vec3, n: Vec3): Vec3 =>
  vec3Sub(v, vec3Scale(n, 2 * vec3Dot(v, n)))

// ----------------------------------------------------------------
// MAT4 (column-major Float32Array)
// ----------------------------------------------------------------

export type Mat4 = Float32Array  // length 16

export const mat4Identity = (): Mat4 => new Float32Array([
  1,0,0,0,
  0,1,0,0,
  0,0,1,0,
  0,0,0,1,
])

export const mat4Multiply = (a: Mat4, b: Mat4): Mat4 => {
  const out = new Float32Array(16)
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0
      for (let k = 0; k < 4; k++) {
        sum += a[i + k * 4]! * b[k + j * 4]!
      }
      out[i + j * 4] = sum
    }
  }
  return out
}

export const mat4Translation = (x: number, y: number, z: number): Mat4 =>
  new Float32Array([
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    x,y,z,1,
  ])

export const mat4Scale = (sx: number, sy: number, sz: number): Mat4 =>
  new Float32Array([
    sx,0,0,0,
    0,sy,0,0,
    0,0,sz,0,
    0,0,0,1,
  ])

export const mat4RotationY = (angle: number): Mat4 => {
  const c = Math.cos(angle), s = Math.sin(angle)
  return new Float32Array([
    c,0,-s,0,
    0,1,0,0,
    s,0,c,0,
    0,0,0,1,
  ])
}

export const mat4Perspective = (
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): Mat4 => {
  const f   = 1 / Math.tan(fovY * 0.5)
  const nf  = 1 / (near - far)
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)*nf, -1,
    0, 0, 2*far*near*nf, 0,
  ])
}

export const mat4LookAt = (
  eye: Vec3,
  target: Vec3,
  up: Vec3,
): Mat4 => {
  const f = vec3Norm(vec3Sub(target, eye))
  const r = vec3Norm(vec3Cross(f, up))
  const u = vec3Cross(r, f)

  return new Float32Array([
    r[0],  u[0], -f[0], 0,
    r[1],  u[1], -f[1], 0,
    r[2],  u[2], -f[2], 0,
    -vec3Dot(r, eye), -vec3Dot(u, eye), vec3Dot(f, eye), 1,
  ])
}

export const mat4TransformVec3 = (m: Mat4, v: Vec3, w = 1): Vec3 => {
  const x = m[0]!*v[0] + m[4]!*v[1] + m[8]!*v[2]  + m[12]!*w
  const y = m[1]!*v[0] + m[5]!*v[1] + m[9]!*v[2]  + m[13]!*w
  const z = m[2]!*v[0] + m[6]!*v[1] + m[10]!*v[2] + m[14]!*w
  const rw= m[3]!*v[0] + m[7]!*v[1] + m[11]!*v[2] + m[15]!*w
  if (Math.abs(rw) > EPSILON && w !== 0) return [x/rw, y/rw, z/rw]
  return [x, y, z]
}

// ----------------------------------------------------------------
// LRU CACHE (простий, для утиліт)
// ----------------------------------------------------------------

export class LRUCache<K, V> {
  private _map = new Map<K, V>()

  constructor(private readonly maxSize: number) {}

  get(key: K): V | undefined {
    if (!this._map.has(key)) return undefined
    const value = this._map.get(key)!
    // Переміщуємо в кінець (найновіший)
    this._map.delete(key)
    this._map.set(key, value)
    return value
  }

  set(key: K, value: V): void {
    if (this._map.has(key)) this._map.delete(key)
    else if (this._map.size >= this.maxSize) {
      // Видаляємо найстаріший (перший у Map)
      this._map.delete(this._map.keys().next().value!)
    }
    this._map.set(key, value)
  }

  has(key: K):    boolean { return this._map.has(key) }
  delete(key: K): boolean { return this._map.delete(key) }
  clear():        void    { this._map.clear() }

  get size(): number { return this._map.size }

  keys():   IterableIterator<K> { return this._map.keys() }
  values(): IterableIterator<V> { return this._map.values() }
  }
