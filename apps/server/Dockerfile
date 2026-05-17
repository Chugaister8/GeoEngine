# GeoEngine — Python Server Dockerfile
# Multi-stage build: builder → production / development

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.4.0

# ================================================================
# Stage 1: Dependencies (uv install)
# ================================================================
FROM python:${PYTHON_VERSION}-slim AS deps

WORKDIR /app

# Встановлення uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Системні залежності для geospatial бібліотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GDAL / rasterio
    libgdal-dev \
    gdal-bin \
    # PROJ
    libproj-dev \
    proj-bin \
    # Загальні
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо workspace конфігурацію
COPY pyproject.toml ./
COPY packages/core-python/pyproject.toml ./packages/core-python/
COPY apps/server/pyproject.toml          ./apps/server/

# Встановлення залежностей (з кешем)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# ================================================================
# Stage 2: Production
# ================================================================
FROM python:${PYTHON_VERSION}-slim AS production

WORKDIR /app

# Системні runtime бібліотеки (без dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    libproj25 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо venv з deps stage
COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /usr/local/bin/uv /usr/local/bin/uv

# Копіюємо код
COPY packages/core-python /app/packages/core-python
COPY apps/server/src      /app/apps/server/src
COPY apps/server/pyproject.toml /app/apps/server/

# PATH для venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Встановлення проекту
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e packages/core-python -e apps/server

# Не-root user для безпеки
RUN useradd --create-home --shell /bin/bash geoengine
RUN chown -R geoengine:geoengine /app
USER geoengine

# Директорії кешу
RUN mkdir -p /home/geoengine/.geoengine/dem_cache \
             /home/geoengine/.geoengine/osm_cache

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn з оптимальними налаштуваннями
CMD ["uvicorn", \
     "apps.server.src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

# ================================================================
# Stage 3: Development (з hot reload)
# ================================================================
FROM production AS development

USER root

# Dev залежності
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install pytest pytest-asyncio httpx

USER geoengine

CMD ["uvicorn", \
     "apps.server.src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/apps/server/src", \
     "--reload-dir", "/app/packages/core-python/geoengine"]
