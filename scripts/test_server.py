"""
GeoEngine — Test Server
Перевіряє що FastAPI сервер відповідає коректно.

Запуск (сервер має бути запущений):
    uvicorn apps.server.src.main:app --port 8000 &
    python scripts/test_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://localhost:8000"


async def check_health() -> None:
    print("🏥 Health check...")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/health", timeout=5)

    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    assert data["status"] == "ok", f"status={data['status']}"
    print(f"   ✅ status=ok  version={data['version']}  "
          f"connections={data['connections']}")


async def check_sources() -> None:
    print("\n📡 Terrain sources...")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/api/terrain/sources", timeout=10)

    assert r.status_code == 200
    sources = r.json()
    assert len(sources) > 0
    ids = [s["id"] for s in sources]
    print(f"   ✅ {len(sources)} джерел: {ids}")


async def check_tile_meta() -> None:
    print("\n🗺  Tile meta (z=9)...")
    # Тайл що містить Карпати
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{BASE}/api/terrain/tile/9/284/178/meta"
            "?source=terrarium",
            timeout=30,
        )

    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ min={data.get('min_elevation',0):.0f}м  "
              f"max={data.get('max_elevation',0):.0f}м  "
              f"resolution={data.get('resolution_m',0):.0f}м/px")
    else:
        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:100]}")


async def check_tile_png() -> None:
    print("\n🖼  Tile PNG...")
    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        r  = await c.get(
            f"{BASE}/api/terrain/tile/9/284/178.png"
            "?source=terrarium&colormap=terrain",
            timeout=30,
        )
        elapsed = time.perf_counter() - t0

    if r.status_code == 200:
        ct = r.headers.get("content-type", "")
        sz = len(r.content)
        assert "image/png" in ct, f"Content-Type: {ct}"
        print(f"   ✅ PNG {sz:,} bytes  "
              f"Content-Type: image/png  "
              f"({elapsed*1000:.0f}мс)")
    else:
        print(f"   ⚠️ HTTP {r.status_code}")


async def check_elevation() -> None:
    print("\n⛰  Elevation batch...")
    points = [[48.16, 24.50], [50.45, 30.52], [46.48, 30.72]]  # Говерла, Київ, Одеса

    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        r  = await c.post(
            f"{BASE}/api/terrain/elevation",
            json={"points": points, "source": "terrarium"},
            timeout=30,
        )
        elapsed = time.perf_counter() - t0

    if r.status_code == 200:
        data   = r.json()
        elevs  = data.get("elevations", [])
        labels = ["Говерла", "Київ", "Одеса"]
        for label, e in zip(labels, elevs):
            status = "✅" if e is not None else "⚠️"
            print(f"   {status} {label}: {e:.0f}м" if e else f"   ⚠️ {label}: None")
        print(f"   ({elapsed*1000:.0f}мс)")
    else:
        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:150]}")


async def check_mesh_api() -> None:
    print("\n🏗  Mesh API (малий bbox)...")
    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        r  = await c.post(
            f"{BASE}/api/terrain/mesh",
            json={
                "west":  24.0, "south": 48.0,
                "east":  24.1, "north": 48.1,
                "source":       "terrarium",
                "max_vertices": 4096,
            },
            timeout=60,
        )
        elapsed = time.perf_counter() - t0

    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ verts={data.get('vertex_count',0):,}  "
              f"tris={data.get('triangle_count',0):,}  "
              f"mem={data.get('memory_bytes',0)//1024}KB  "
              f"({elapsed*1000:.0f}мс)")
        bufs = data.get("buffers", {})
        print(f"   ✅ buffers: {list(bufs.keys())}")
    else:
        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:150]}")


async def check_mesh_too_large() -> None:
    print("\n🚫 Mesh — bbox занадто великий...")
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{BASE}/api/terrain/mesh",
            json={"west": 0, "south": 0, "east": 10, "north": 10, "source": "terrarium"},
            timeout=10,
        )

    if r.status_code == 400:
        print(f"   ✅ HTTP 400 (очікувано): {r.json().get('detail','')[:80]}")
    else:
        print(f"   ⚠️ HTTP {r.status_code} (очікувалось 400)")


async def check_websocket() -> None:
    print("\n🔌 WebSocket...")
    try:
        import websockets

        async with websockets.connect("ws://localhost:8000/ws") as ws:
            # Ping
            msg = json.dumps({
                "type": "ping",
                "id":   "test-ping-001",
                "timestamp": 0,
                "payload": {},
            })
            await ws.send(msg)
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert resp["type"] == "pong", f"type={resp['type']}"
            print("   ✅ ping → pong OK")

            # Request tile
            msg2 = json.dumps({
                "type": "request_tile",
                "id":   "test-tile-001",
                "timestamp": 0,
                "payload": {
                    "tile":   {"x": 284, "y": 178, "z": 9},
                    "source": "terrarium",
                    "max_vertices": 1024,
                },
            })
            t0 = time.perf_counter()
            await ws.send(msg2)
            resp2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            elapsed = time.perf_counter() - t0

            if resp2["type"] == "response_tile":
                p = resp2["payload"]
                print(f"   ✅ response_tile: verts={p.get('vertex_count',0):,}  "
                      f"tris={p.get('triangle_count',0):,}  "
                      f"({elapsed*1000:.0f}мс)")
                print(f"   ✅ request_id={resp2.get('request_id','')} "
                      f"== sent id: {resp2.get('request_id')=='test-tile-001'}")
            elif resp2["type"] == "error":
                print(f"   ⚠️ Error: {resp2['payload']}")
            else:
                print(f"   ⚠️ Unexpected: {resp2['type']}")

    except ImportError:
        print("   ⚠️ websockets не встановлено: pip install websockets")
    except Exception as e:
        print(f"   ❌ {e}")


async def check_cache() -> None:
    print("\n💾 Cache stats...")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/api/terrain/cache/stats", timeout=10)

    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ files={data.get('files',0)}  "
              f"size={data.get('size_mb',0)}MB")
    else:
        print(f"   ⚠️ HTTP {r.status_code}")


async def main() -> None:
    print("=" * 60)
    print("◈  GeoEngine — Server Test")
    print(f"   URL: {BASE}")
    print("=" * 60)

    # Перевірити чи сервер запущений
    try:
        async with httpx.AsyncClient() as c:
            await c.get(f"{BASE}/health", timeout=3)
    except Exception:
        print(f"\n❌ Сервер не відповідає на {BASE}")
        print("   Запусти спочатку:")
        print("   uvicorn apps.server.src.main:app --port 8000 --reload")
        sys.exit(1)

    tests = [
        check_health,
        check_sources,
        check_tile_meta,
        check_tile_png,
        check_elevation,
        check_mesh_api,
        check_mesh_too_large,
        check_websocket,
        check_cache,
    ]

    errors = 0
    for test in tests:
        try:
            await test()
        except AssertionError as e:
            print(f"   ❌ Assertion: {e}")
            errors += 1
        except Exception as e:
            print(f"   ❌ Exception: {type(e).__name__}: {e}")
            errors += 1

    print("\n" + "=" * 60)
    if errors == 0:
        print("✅ ВСІ СЕРВЕР ТЕСТИ ПРОЙШЛИ")
        print("\n🎉 Наступний крок:")
        print("   Відкрий http://localhost:3000")
        print("   Перевір 3D рендеринг у браузері")
    else:
        print(f"⚠️  {errors} тестів не пройшли")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
