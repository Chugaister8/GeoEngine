# GeoEngine — Next.js Viewer Dockerfile
# Multi-stage: deps → builder → production / development

ARG NODE_VERSION=20
ARG PNPM_VERSION=9.0.0

# ================================================================
# Stage 1: Base (pnpm setup)
# ================================================================
FROM node:${NODE_VERSION}-alpine AS base

RUN corepack enable && corepack prepare pnpm@${PNPM_VERSION} --activate
WORKDIR /app

# ================================================================
# Stage 2: Dependencies
# ================================================================
FROM base AS deps

# Копіюємо workspace маніфести
COPY package.json pnpm-workspace.yaml ./
COPY packages/shared-types/package.json   ./packages/shared-types/
COPY packages/core-js/package.json        ./packages/core-js/
COPY apps/viewer/package.json             ./apps/viewer/

# Встановлення всіх deps (з кешем)
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

# ================================================================
# Stage 3: Builder
# ================================================================
FROM deps AS builder

ARG NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
ARG NEXT_PUBLIC_API_URL=http://localhost:8000

ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1

# Копіюємо весь код
COPY packages/shared-types ./packages/shared-types
COPY packages/core-js      ./packages/core-js
COPY apps/viewer           ./apps/viewer
COPY turbo.json            ./

# Build shared packages спочатку
RUN pnpm --filter @geoengine/shared-types build
RUN pnpm --filter @geoengine/core-js build

# Build Next.js
RUN pnpm --filter @geoengine/viewer build

# ================================================================
# Stage 4: Production runner
# ================================================================
FROM node:${NODE_VERSION}-alpine AS production

RUN corepack enable && corepack prepare pnpm@9.0.0 --activate
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Копіюємо тільки що потрібно для runtime
COPY --from=builder /app/apps/viewer/.next/standalone ./
COPY --from=builder /app/apps/viewer/.next/static     ./apps/viewer/.next/static
COPY --from=builder /app/apps/viewer/public           ./apps/viewer/public

# Не-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser  --system --uid 1001 nextjs
USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD wget -qO- http://localhost:3000 || exit 1

CMD ["node", "apps/viewer/server.js"]

# ================================================================
# Stage 5: Development (hot reload)
# ================================================================
FROM deps AS development

ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

COPY packages/shared-types ./packages/shared-types
COPY packages/core-js      ./packages/core-js
COPY apps/viewer           ./apps/viewer
COPY turbo.json            ./

EXPOSE 3000

CMD ["pnpm", "--filter", "@geoengine/viewer", "dev"]
