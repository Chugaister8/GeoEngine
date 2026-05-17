# GeoEngine — Developer Commands
# Використання: make <target>

.PHONY: help dev prod build test lint typecheck clean logs shell

# ----------------------------------------------------------------
# Кольори для виводу
# ----------------------------------------------------------------
CYAN   := \033[0;36m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
RESET  := \033[0m

# ----------------------------------------------------------------
# HELP
# ----------------------------------------------------------------
help:
	@echo ""
	@echo "$(CYAN)GeoEngine — Developer Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Запуск:$(RESET)"
	@echo "  make dev          — запустити в режимі розробки (hot reload)"
	@echo "  make prod         — запустити в production режимі"
	@echo "  make stop         — зупинити всі сервіси"
	@echo ""
	@echo "$(GREEN)Збірка:$(RESET)"
	@echo "  make build        — зібрати всі Docker образи"
	@echo "  make build-server — зібрати тільки server"
	@echo "  make build-viewer — зібрати тільки viewer"
	@echo ""
	@echo "$(GREEN)Тести:$(RESET)"
	@echo "  make test         — запустити всі тести"
	@echo "  make test-py      — Python тести"
	@echo "  make test-js      — JavaScript тести"
	@echo ""
	@echo "$(GREEN)Якість коду:$(RESET)"
	@echo "  make lint         — ruff + eslint"
	@echo "  make typecheck    — mypy + tsc"
	@echo "  make fmt          — форматування коду"
	@echo ""
	@echo "$(GREEN)Утиліти:$(RESET)"
	@echo "  make logs         — показати логи всіх сервісів"
	@echo "  make logs-server  — логи тільки сервера"
	@echo "  make shell-server — bash у server контейнері"
	@echo "  make clean        — видалити volumes та кеш"
	@echo "  make setup        — початкове налаштування"
	@echo ""

# ----------------------------------------------------------------
# ЗАПУСК
# ----------------------------------------------------------------
dev:
	@echo "$(CYAN)Запуск в dev режимі...$(RESET)"
	BUILD_TARGET=development docker compose up --build

dev-bg:
	@echo "$(CYAN)Запуск в dev режимі (background)...$(RESET)"
	BUILD_TARGET=development docker compose up --build -d

prod:
	@echo "$(CYAN)Запуск в production режимі...$(RESET)"
	BUILD_TARGET=production docker compose --profile prod up --build -d

stop:
	@echo "$(YELLOW)Зупинка сервісів...$(RESET)"
	docker compose down

restart: stop dev-bg

# ----------------------------------------------------------------
# ЗБІРКА
# ----------------------------------------------------------------
build:
	@echo "$(CYAN)Збірка всіх образів...$(RESET)"
	docker compose build

build-server:
	docker compose build server

build-viewer:
	docker compose build viewer

# ----------------------------------------------------------------
# ТЕСТИ
# ----------------------------------------------------------------
test: test-py test-js
	@echo "$(GREEN)Всі тести пройшли!$(RESET)"

test-py:
	@echo "$(CYAN)Python тести...$(RESET)"
	uv run pytest tests/python/ -v --cov=packages/core-python/geoengine

test-js:
	@echo "$(CYAN)JavaScript тести...$(RESET)"
	pnpm --filter @geoengine/core-js test

test-integration:
	@echo "$(CYAN)Integration тести...$(RESET)"
	docker compose up -d server redis
	sleep 15
	curl -f http://localhost:8000/health
	curl -f http://localhost:8000/api/terrain/sources
	docker compose down

# ----------------------------------------------------------------
# ЯКІСТЬ КОДУ
# ----------------------------------------------------------------
lint:
	@echo "$(CYAN)Linting...$(RESET)"
	uv run ruff check packages/core-python apps/server
	pnpm --filter @geoengine/core-js lint 2>/dev/null || true

typecheck:
	@echo "$(CYAN)Type checking...$(RESET)"
	uv run mypy packages/core-python/geoengine --ignore-missing-imports
	pnpm --filter @geoengine/shared-types typecheck
	pnpm --filter @geoengine/core-js typecheck
	pnpm --filter @geoengine/viewer typecheck

fmt:
	@echo "$(CYAN)Форматування...$(RESET)"
	uv run ruff format packages/core-python apps/server
	uv run ruff check --fix packages/core-python apps/server
	pnpm --filter @geoengine/core-js format 2>/dev/null || true

# ----------------------------------------------------------------
# ЛОГИ ТА SHELL
# ----------------------------------------------------------------
logs:
	docker compose logs -f

logs-server:
	docker compose logs -f server

logs-viewer:
	docker compose logs -f viewer

logs-redis:
	docker compose logs -f redis

shell-server:
	docker compose exec server bash

shell-viewer:
	docker compose exec viewer sh

# ----------------------------------------------------------------
# УТИЛІТИ
# ----------------------------------------------------------------
setup:
	@echo "$(CYAN)Початкове налаштування...$(RESET)"
	@test -f .env || (cp .env.example .env && echo "$(GREEN)Створено .env з .env.example$(RESET)")
	@echo "$(YELLOW)Відредагуй .env та додай API ключі$(RESET)"
	uv sync --all-packages
	pnpm install

clean:
	@echo "$(YELLOW)Очищення...$(RESET)"
	docker compose down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .next -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Очищено$(RESET)"

clean-cache:
	@echo "$(YELLOW)Очищення кешу DEM та OSM...$(RESET)"
	docker compose exec server python -c \
		"from geoengine.dem.sources import DEMSourceManager; \
		 m = DEMSourceManager(); print(f'Видалено {m.clear_cache()} файлів')"

status:
	@echo "$(CYAN)Статус сервісів:$(RESET)"
	docker compose ps
	@echo ""
	@echo "$(CYAN)Health checks:$(RESET)"
	@curl -sf http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "Server: недоступний"
	@curl -sf http://localhost:3000 > /dev/null && echo "Viewer: OK" || echo "Viewer: недоступний"

# ----------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------
demo-carpathians:
	@echo "$(CYAN)Завантаження демо даних (Карпати)...$(RESET)"
	curl -X POST http://localhost:8000/api/terrain/mesh \
		-H "Content-Type: application/json" \
		-d '{"west":23.0,"south":48.0,"east":25.0,"north":49.0,"source":"copernicus25"}' \
		| python -m json.tool | head -20

demo-buildings:
	@echo "$(CYAN)Завантаження будівель (Ужгород)...$(RESET)"
	curl -X POST http://localhost:8000/api/osm/buildings \
		-H "Content-Type: application/json" \
		-d '{"west":22.28,"south":48.60,"east":22.32,"north":48.63}' \
		| python -m json.tool | head -20
