/**
 * GeoEngine — Головна сторінка вьюера
 * Next.js 14 App Router page.
 */

"use client"

import { useCallback, useState } from "react"
import { GeoCanvas }  from "../components/GeoCanvas"
import { LayerPanel, DEFAULT_LAYERS, type Layer } from "../components/LayerPanel"
import { Minimap }    from "../components/Minimap"

// ----------------------------------------------------------------
// КОНФІГУРАЦІЯ
// ----------------------------------------------------------------

const INITIAL_CAMERA = {
  lat:      48.15,    // Карпати
  lon:      24.50,
  altitude: 12_000,
}

const SERVER_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws"

// ----------------------------------------------------------------
// СТОРІНКА
// ----------------------------------------------------------------

export default function ViewerPage() {
  const [layers,    setLayers]    = useState<Layer[]>(DEFAULT_LAYERS)
  const [cameraPos, setCameraPos] = useState(INITIAL_CAMERA)

  // ---- Layer management ----

  const handleToggleLayer = useCallback((id: string): void => {
    setLayers(prev => prev.map(l =>
      l.id === id ? { ...l, visible: !l.visible } : l
    ))
  }, [])

  const handleOpacityChange = useCallback((id: string, opacity: number): void => {
    setLayers(prev => prev.map(l =>
      l.id === id ? { ...l, opacity } : l
    ))
  }, [])

  // ---- Рендер ----

  return (
    <main className="w-screen h-screen overflow-hidden bg-black relative">

      {/* Головний 3D вьюпорт */}
      <GeoCanvas
        serverUrl     = {SERVER_URL}
        initialCamera = {INITIAL_CAMERA}
        renderer      = "auto"
        showStats     = {true}
        showCompass   = {true}
        showCoords    = {true}
        showToolbar   = {true}
        height        = "100vh"
        debug         = {process.env.NODE_ENV === "development"}
        onReady={() => {
          console.log("[GeoEngine] Рендерер готовий 🚀")
        }}
        onError={(err) => {
          console.error("[GeoEngine] Помилка:", err)
        }}
      />

      {/* Layer Panel */}
      <LayerPanel
        layers    = {layers}
        onToggle  = {handleToggleLayer}
        onOpacity = {handleOpacityChange}
      />

      {/* Minimap */}
      <div className="absolute bottom-4 left-3 z-10">
        <Minimap
          cameraLat  = {cameraPos.lat}
          cameraLon  = {cameraPos.lon}
          viewRadiusM = {cameraPos.altitude * 2}
          className  = "w-[160px] h-[160px]"
        />
      </div>

      {/* Лого */}
      <div className="
        absolute top-3 left-1/2 -translate-x-1/2 z-10
        pointer-events-none select-none
      ">
        <div className="
          font-mono text-xs tracking-[0.3em]
          text-white/30 uppercase
        ">
          ◈ GeoEngine
        </div>
      </div>

    </main>
  )
}
