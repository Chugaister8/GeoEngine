/**
 * GeoEngine — Object Pool
 * Переиспользование Three.js об'єктів для уникнення GC пауз.
 *
 * Використовується для: Vector3, Matrix4, Euler,
 * BufferGeometry патчів, тимчасових масивів.
 */

export class ObjectPool<T> {
  /**
   * Generic object pool.
   *
   * Мінімізує GC pressure у render loop шляхом переиспользування об'єктів.
   *
   * Usage:
   *   const vec3Pool = new ObjectPool(
   *     () => new THREE.Vector3(),
   *     (v) => v.set(0, 0, 0),
   *     32,
   *   )
   *   const v = vec3Pool.acquire()
   *   // ... використання
   *   vec3Pool.release(v)
   */

  private _pool:    T[] = []
  private _active:  Set<T> = new Set()
  private _created: number = 0

  constructor(
    private readonly _factory:  () => T,
    private readonly _reset:    (obj: T) => void,
    private readonly _maxSize:  number = 64,
  ) {}

  /**
   * Отримати об'єкт з пулу або створити новий.
   */
  acquire(): T {
    let obj: T

    if (this._pool.length > 0) {
      obj = this._pool.pop()!
    } else {
      obj = this._factory()
      this._created++
    }

    this._active.add(obj)
    return obj
  }

  /**
   * Повернути об'єкт у пул.
   * Скидає стан через _reset функцію.
   */
  release(obj: T): void {
    if (!this._active.has(obj)) return
    this._active.delete(obj)

    if (this._pool.length < this._maxSize) {
      this._reset(obj)
      this._pool.push(obj)
    }
    // Якщо пул повний — просто відпускаємо об'єкт (GC забере)
  }

  /**
   * Повернути всі активні об'єкти.
   * Корисно після кожного кадру.
   */
  releaseAll(): void {
    for (const obj of this._active) {
      if (this._pool.length < this._maxSize) {
        this._reset(obj)
        this._pool.push(obj)
      }
    }
    this._active.clear()
  }

  get poolSize():   number { return this._pool.length }
  get activeCount(): number { return this._active.size }
  get totalCreated(): number { return this._created }

  clear(): void {
    this._pool.length = 0
    this._active.clear()
  }
}


/**
 * Float32Array pool — для тимчасових GPU буферів.
 * Групує за розміром щоб уникнути алокацій.
 */
export class Float32ArrayPool {
  private _buckets: Map<number, Float32Array[]> = new Map()

  acquire(size: number): Float32Array {
    const bucket = this._buckets.get(size)
    if (bucket && bucket.length > 0) {
      return bucket.pop()!
    }
    return new Float32Array(size)
  }

  release(arr: Float32Array): void {
    const bucket = this._buckets.get(arr.length)
    if (bucket) {
      if (bucket.length < 8) bucket.push(arr)
    } else {
      this._buckets.set(arr.length, [arr])
    }
  }

  clear(): void { this._buckets.clear() }
}
