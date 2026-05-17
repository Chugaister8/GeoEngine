import { defineConfig } from "vitest/config"
import { resolve } from "path"

export default defineConfig({
  test: {
    globals:     true,
    environment: "jsdom",
    coverage: {
      provider:  "v8",
      reporter:  ["text", "lcov", "html"],
      include:   ["src/**/*.ts"],
      exclude:   ["src/**/*.test.ts", "src/shaders/**"],
    },
    setupFiles: ["./tests/setup.ts"],
  },
  resolve: {
    alias: {
      "@geoengine/shared-types": resolve(__dirname, "../shared-types/src/geo"),
    },
  },
})
