import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  // Three.js потребує transpile
  transpilePackages: [
    "@geoengine/core-js",
    "@geoengine/shared-types",
  ],

  webpack: (config, { isServer }) => {
    // Three.js WebGPU шейдери — raw imports
    config.module.rules.push({
      test:   /\.wgsl$/,
      type:   "asset/source",
    })

    // Ігнорувати node-only модулі на клієнті
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs:   false,
        path: false,
      }
    }

    return config
  },

  // Дозволити WebGPU заголовки
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy",   value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy",  value: "require-corp" },
        ],
      },
    ]
  },
}

export default nextConfig
