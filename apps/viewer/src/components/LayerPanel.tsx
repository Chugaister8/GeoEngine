/**
 * GeoEngine — LayerPanel
 * Панель управління шарами (terrain, OSM, satellite, analysis).
 */

"use client"

import { useState, type ReactNode } from "react"

// ----------------------------------------------------------------
// ТИПИ
// ----------------------------------------------------------------

export interface Layer {
  id:      string
  name:    string
  icon:    string
  visible: boolean
  opacity: number
  type:    "terrain" | "satellite" | "osm" | "analysis" | "custom"
}

export interface LayerPanelProps {
  layers:        Layer[]
  onToggle:      (id: string) => void
  onOpacity:     (id: string, opacity: number) => void
  onReorder?:    (fromIdx: number, toIdx: number) => void
}

// ----------------------------------------------------------------
// DEFAULT LAYERS
// ----------------------------------------------------------------

export const DEFAULT_LAYERS: Layer[] = [
  {
    id: "terrain",  name: "Рельєф (DEM)",
    icon: "⛰",  visible: true,  opacity: 1.0, type: "terrain",
  },
  {
    id: "satellite", name: "Супутник",
    icon: "🛰",  visible: true,  opacity: 1.0, type: "satellite",
  },
  {
    id: "osm_buildings", name: "Будівлі (OSM)",
    icon: "🏢",  visible: false, opacity: 0.9, type: "osm",
  },
  {
    id: "osm_roads", name: "Дороги (OSM)",
    icon: "🛣",  visible: false, opacity: 0.8, type: "osm",
  },
  {
    id: "slope", name: "Крутизна схилів",
    icon: "📐", visible: false, opacity: 0.7, type: "analysis",
  },
  {
    id: "hillshade", name: "Тіньовий рельєф",
    icon: "☀",  visible: false, opacity: 0.6, type: "analysis",
  },
  {
    id: "contours", name: "Ізолінії (100м)",
    icon: "〰",  visible: false, opacity: 0.8, type: "analysis",
  },
]

// ----------------------------------------------------------------
// КОМПОНЕНТ
// ----------------------------------------------------------------

export function LayerPanel({
  layers,
  onToggle,
  onOpacity,
}: LayerPanelProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className={`
      absolute left-3 top-1/2 -translate-y-1/2 z-10
      bg-black/70 backdrop-blur border border-white/15
      rounded-lg overflow-hidden
      transition-all duration-200
      ${collapsed ? "w-10" : "w-52"}
    `}>
      {/* Header */}
      <button
        onClick={() => setCollapsed(v => !v)}
        className="
          w-full flex items-center justify-between
          px-3 py-2.5
          text-white/70 hover:text-white
          border-b border-white/10
          transition-colors
        "
      >
        {!collapsed && (
          <span className="text-xs font-mono uppercase tracking-wider">
            Шари
          </span>
        )}
        <span className="text-sm ml-auto">
          {collapsed ? "≡" : "✕"}
        </span>
      </button>

      {/* Layers list */}
      {!collapsed && (
        <div className="py-1">
          {layers.map(layer => (
            <LayerRow
              key={layer.id}
              layer={layer}
              onToggle={() => onToggle(layer.id)}
              onOpacity={v => onOpacity(layer.id, v)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ---- Layer Row ----

function LayerRow({
  layer,
  onToggle,
  onOpacity,
}: {
  layer:     Layer
  onToggle:  () => void
  onOpacity: (v: number) => void
}) {
  const [showSlider, setShowSlider] = useState(false)

  return (
    <div className={`
      group px-2 py-1.5
      ${layer.visible ? "" : "opacity-50"}
      hover:bg-white/5 transition-colors
    `}>
      {/* Main row */}
      <div className="flex items-center gap-2">
        {/* Visibility toggle */}
        <button
          onClick={onToggle}
          className="
            w-5 h-5 flex items-center justify-center
            text-white/60 hover:text-white
            transition-colors flex-shrink-0
          "
        >
          {layer.visible ? "◉" : "○"}
        </button>

        {/* Icon + Name */}
        <span className="text-sm">{layer.icon}</span>
        <span className="text-xs font-mono text-white/80 flex-1 truncate">
          {layer.name}
        </span>

        {/* Opacity button */}
        <button
          onClick={() => setShowSlider(v => !v)}
          className="
            opacity-0 group-hover:opacity-100
            text-white/40 hover:text-white/80
            text-xs transition-all
          "
          title="Прозорість"
        >
          ◐
        </button>
      </div>

      {/* Opacity slider */}
      {showSlider && (
        <div className="mt-1.5 px-7">
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={layer.opacity}
            onChange={e => onOpacity(parseFloat(e.target.value))}
            className="w-full h-1 accent-blue-400 cursor-pointer"
          />
          <div className="flex justify-between text-white/30 text-[10px] font-mono mt-0.5">
            <span>0%</span>
            <span>{Math.round(layer.opacity * 100)}%</span>
            <span>100%</span>
          </div>
        </div>
      )}
    </div>
  )
}
