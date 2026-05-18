#!/usr/bin/env bash
# GeoEngine — Quick Start Script
# Крок за кроком запускає весь стек

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   ◈  GeoEngine — Quick Start         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
echo ""

# ── Крок 1: Перевірка залежностей ───────────────────────────────
echo -e "${YELLOW}[1/6] Перевірка залежностей...${NC}"

command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ python3 не знайдено${NC}"; exit 1; }
command -v uv      >/dev/null 2>&1 || { echo -e "${RED}❌ uv не знайдено. Встанови: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"; exit 1; }
command -v pnpm    >/dev/null 2>&1 || { echo -e "${RED}❌ pnpm не знайдено. Встанови: npm i -g pnpm${NC}"; exit 1; }

PY_VER=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}   ✅ Python $PY_VER${NC}"
echo -e "${GREEN}   ✅ uv $(uv --version)${NC}"
echo -e "${GREEN}   ✅ pnpm $(pnpm --version)${NC}"

# ── Крок 2: Встановлення залежностей ────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Встановлення Python залежностей...${NC}"
uv sync --all-packages 2>&1 | grep -E "(Installed|error|ERROR)" || true
echo -e "${GREEN}   ✅ Python deps OK${NC}"

echo ""
echo -e "${YELLOW}[3/6] Встановлення JS залежностей...${NC}"
pnpm install --frozen-lockfile 2>&1 | tail -3
echo -e "${GREEN}   ✅ JS deps OK${NC}"

# ── Крок 3: Seed дані ────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Завантаження seed даних (Карпати)...${NC}"
if [ -f "data/dem/carpathians.tif" ]; then
    echo -e "${GREEN}   ✅ Seed дані вже є: data/dem/carpathians.tif${NC}"
else
    echo "   Завантаження з AWS Terrarium (без API ключа)..."
    uv run python scripts/seed_data.py
fi

# ── Крок 4: Python pipeline test ─────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Тест Python pipeline...${NC}"
uv run python scripts/test_mesh.py

# ── Крок 5: Запуск стека ─────────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Запуск GeoEngine стека...${NC}"
echo ""

# Вибір режиму
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "Доступні варіанти:"
    echo "  1) Docker Compose (рекомендовано)"
    echo "  2) Без Docker (uvicorn + pnpm dev)"
    read -rp "Вибір [1/2]: " choice
else
    choice=2
    echo "Docker недоступний, запускаємо без нього."
fi

if [ "$choice" = "1" ]; then
    # Docker
    echo -e "${CYAN}   Запуск через Docker Compose...${NC}"
    cp .env.example .env 2>/dev/null || true
    docker compose up --build -d
    echo ""
    sleep 5
    curl -sf http://localhost:8000/health | python3 -m json.tool
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ GeoEngine запущено!               ║${NC}"
    echo -e "${GREEN}║                                      ║${NC}"
    echo -e "${GREEN}║  🌍 Вьюер:  http://localhost:3000    ║${NC}"
    echo -e "${GREEN}║  📡 API:    http://localhost:8000    ║${NC}"
    echo -e "${GREEN}║  📖 Docs:   http://localhost:8000/docs║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
else
    # Без Docker
    echo -e "${CYAN}   Запуск сервера (background)...${NC}"
    uv run uvicorn apps.server.src.main:app \
        --host 0.0.0.0 --port 8000 --reload \
        --log-level warning &
    SERVER_PID=$!
    echo "   Server PID: $SERVER_PID"

    sleep 3
    echo ""
    echo -e "${CYAN}   Тест сервера...${NC}"
    uv run python scripts/test_server.py

    echo ""
    echo -e "${CYAN}   Запуск вьюера...${NC}"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ GeoEngine запускається!           ║${NC}"
    echo -e "${GREEN}║                                      ║${NC}"
    echo -e "${GREEN}║  📡 API:  http://localhost:8000      ║${NC}"
    echo -e "${GREEN}║  🌍 Вьюер буде на: localhost:3000   ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo "   (Ctrl+C щоб зупинити сервер)"

    pnpm --filter @geoengine/viewer dev
fi
