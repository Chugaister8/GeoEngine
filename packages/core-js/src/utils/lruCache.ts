/**
 * GeoEngine — LRU Cache
 * Generic Least Recently Used cache.
 * Використовується для: тайлів, геометрій, текстур, DEM даних.
 */

export interface LRUCacheOptions<K, V> {
  maxSize:    number
  onEvict?:   (key: K, value: V) => void   // викликається при видаленні
  ttlMs?:     number                        // TTL в мілісекундах (0 = без TTL)
}

interface CacheEntry<V> {
  value:     V
  lastUsed:  number    // performance.now()
  createdAt: number
}

export class LRUCache<K, V> {
  /**
   * LRU (Least Recently Used) cache з опційним TTL.
   *
   * Алгоритм: Map зберігає порядок вставки;
   * при доступі — переміщуємо ключ в кінець.
   * При переповненні — видаляємо перший (найстаріший).
   *
   * O(1) get/set завдяки Map.
   */

  private _map:  Map<K, CacheEntry<V>> = new Map()
  private _opts: Required<LRUCacheOptions<K, V>>

  // Статистика
  private _hits   = 0
  private _misses = 0

  constructor(options: LRUCacheOptions<K, V>) {
    this._opts = {
      maxSize: options.maxSize,
      onEvict: options.onEvict ?? (() => {}),
      ttlMs:   options.ttlMs   ?? 0,
    }
  }

  get(key: K): V | undefined {
    const entry = this._map.get(key)

    if (!entry) {
      this._misses++
      return undefined
    }

    // TTL перевірка
    if (this._opts.ttlMs > 0) {
      if (performance.now() - entry.createdAt > this._opts.ttlMs) {
        this._map.delete(key)
        this._opts.onEvict(key, entry.value)
        this._misses++
        return undefined
      }
    }

    // Переміщуємо в кінець (LRU оновлення)
    entry.lastUsed = performance.now()
    this._map.delete(key)
    this._map.set(key, entry)

    this._hits++
    return entry.value
  }

  set(key: K, value: V): void {
    // Якщо ключ вже є — видаляємо щоб оновити порядок
    if (this._map.has(key)) {
      this._map.delete(key)
    } else if (this._map.size >= this._opts.maxSize) {
      // Evict LRU (перший у Map)
      const firstKey = this._map.keys().next().value
      if (firstKey !== undefined) {
        const evicted = this._map.get(firstKey)!
        this._map.delete(firstKey)
        this._opts.onEvict(firstKey, evicted.value)
      }
    }

    this._map.set(key, {
      value,
      lastUsed:  performance.now(),
      createdAt: performance.now(),
    })
  }

  has(key: K): boolean {
    if (!this._map.has(key)) return false
    // TTL перевірка без оновлення LRU
    if (this._opts.ttlMs > 0) {
      const entry = this._map.get(key)!
      if (performance.now() - entry.createdAt > this._opts.ttlMs) {
        this._map.delete(key)
        this._opts.onEvict(key, entry.value)
        return false
      }
    }
    return true
  }

  delete(key: K): boolean {
    const entry = this._map.get(key)
    if (!entry) return false
    this._map.delete(key)
    this._opts.onEvict(key, entry.value)
    return true
  }

  clear(): void {
    for (const [key, entry] of this._map) {
      this._opts.onEvict(key, entry.value)
    }
    this._map.clear()
  }

  /**
   * Видалити всі записи з простроченим TTL.
   * Рекомендується викликати раз на хвилину.
   */
  evictExpired(): number {
    if (this._opts.ttlMs <= 0) return 0
    let count = 0
    const now = performance.now()
    for (const [key, entry] of this._map) {
      if (now - entry.createdAt > this._opts.ttlMs) {
        this._map.delete(key)
        this._opts.onEvict(key, entry.value)
        count++
      }
    }
    return count
  }

  get size(): number { return this._map.size }

  get stats() {
    const total = this._hits + this._misses
    return {
      size:     this._map.size,
      maxSize:  this._opts.maxSize,
      hits:     this._hits,
      misses:   this._misses,
      hitRate:  total > 0 ? this._hits / total : 0,
    }
  }

  keys():   IterableIterator<K> { return this._map.keys() }
  values(): IterableIterator<V> {
    const entries = this._map.values()
    return (function* () {
      for (const e of entries) yield e.value
    })()
  }

  entries(): Array<[K, V]> {
    return [...this._map.entries()].map(([k, e]) => [k, e.value])
  }
}
