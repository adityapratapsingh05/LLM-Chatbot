.PHONY: up down logs test clean build shell-backend

# ── Primary commands ──────────────────────────────────────────────────────────

## Start all services (build first)
up:
	docker compose up --build -d
	@echo ""
	@echo "✅ Services started:"
	@echo "   Backend:  http://localhost:8000"
	@echo "   Frontend: http://localhost:3000"
	@echo "   Health:   http://localhost:8000/health"

## Stop and remove containers (keep volumes)
down:
	docker compose down

## Show logs (all services, follow)
logs:
	docker compose logs -f

## Show backend logs only
logs-backend:
	docker compose logs -f backend

## Show frontend logs only
logs-frontend:
	docker compose logs -f frontend

## Run backend tests (requires a venv or Docker)
test:
	docker compose run --rm --no-deps \
	  -e LLM_API_KEYS=fake_key_1,fake_key_2 \
	  -e DATABASE_URL=sqlite:////tmp/test.db \
	  -e CACHE_DIR=/tmp/test_cache \
	  -e ALLOWED_ORIGINS=http://localhost:3000 \
	  -e ENABLE_FAILURE_SIMULATION=true \
	  backend \
	  sh -c "pip install pytest pytest-asyncio anyio httpx --quiet && pytest -v"

## Run tests locally (needs active venv in backend/)
test-local:
	cd backend && python -m pytest -v

## Stop and remove containers AND volumes (WARNING: deletes DB!)
clean:
	docker compose down -v
	@echo "⚠️  Volumes deleted – database history erased."

## Rebuild without cache
rebuild:
	docker compose build --no-cache
	docker compose up -d

## Health check
health:
	curl -s http://localhost:8000/health | python -m json.tool
