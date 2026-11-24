# LLM Chatbot with Key Failover and Response Caching

A production-ready multi-turn chatbot with automatic LLM key failover and intelligent response caching.

## Live URLs

| Resource | URL |
|---|---|
| **Frontend** | https://llm-chatbot-gamma.vercel.app |
| **Backend** | https://llm-chatbot-g6wl.onrender.com |
| **Health** | https://llm-chatbot-g6wl.onrender.com/health |
| **Repo** | https://github.com/adityapratapsingh05/LLM-Chatbot |
| **Demo Video** | *To be recorded* |

---

## Run locally

### With Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/adityapratapsingh05/LLM-Chatbot.git
cd LLM-Chatbot

# 2. Create .env from example and fill in your keys
cp .env.example .env
# Edit .env – paste at least 2 real Gemini API keys into LLM_API_KEYS

# 3. Start everything
make up

# Services:
#   Backend:  http://localhost:8000
#   Frontend: http://localhost:3000
#   Health:   http://localhost:8000/health
```

### Without Docker

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export LLM_API_KEYS=your_key_1,your_key_2
export DATABASE_URL=sqlite:///./local.db
export CACHE_DIR=./local_cache
export ALLOWED_ORIGINS=http://localhost:5173
export ENABLE_FAILURE_SIMULATION=true
export LITELLM_LOCAL_MODEL_COST_MAP=1
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
# Opens at http://localhost:5173
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEYS` | ✅ yes (≥2) | — | Comma-separated API keys; assigned ids key_1, key_2, … |
| `DATABASE_URL` | — | `sqlite:////data/app.db` | SQLite path (file on volume) |
| `CACHE_DIR` | — | `/data/llm_cache` | Disk cache directory |
| `ALLOWED_ORIGINS` | — | `http://localhost:3000` | Comma-separated CORS origins |
| `ENABLE_FAILURE_SIMULATION` | — | `false` | `true` enables X-Simulate-Failure header |
| `LLM_MODEL` | — | `gemini/gemini-2.0-flash` | LiteLLM model string |
| `DEFAULT_SYSTEM_PROMPT` | — | `You are a helpful assistant.` | Default system prompt |
| `PORT` | — | `8000` | Backend port (honored by uvicorn) |
| `VITE_API_BASE_URL` | — | `http://localhost:8000` | Frontend build-time backend URL |
| `LITELLM_LOCAL_MODEL_COST_MAP` | — | `1` | Prevents litellm fetching cost map on startup (always set to 1) |

---

## Design decisions

### Key Failover

The key pool is parsed from `LLM_API_KEYS` (comma-separated, stripped, empties dropped). Keys are assigned stable ids `key_1`, `key_2`, … in declaration order. The first call always starts with `key_1`.

**Errors that trigger failover:** `RateLimitError` (HTTP 429), `ServiceUnavailableError` (503), `InternalServerError` (5xx), `APIConnectionError`, `Timeout`, and `AuthenticationError`/`BadRequestError` (treat broken/invalid keys as failed too). Each key has `num_retries=0` and a 30-second timeout so the system fails fast and moves to the next key immediately.

If every key fails, the endpoint returns HTTP 503 `{"detail":"all providers unavailable"}` and **nothing** is written to the database (atomicity guarantee).

### Caching

LiteLLM's built-in disk cache (`litellm.cache = Cache(type="disk", disk_cache_dir=CACHE_DIR)`) is used. The cache key is **model + full message list** (system prompt + history + new message), computed deterministically. The API key is **excluded** from the cache key — if `key_1` calls the LLM and fills the cache, `key_2` on a later request still gets a hit. This is correct behavior: the reply is the same regardless of which key was used.

**Temperature is fixed at 0** so responses are deterministic and cache hits produce identical content.

A cache hit returns `resolution_source: "LLM_CACHE"`, `key_id: null`, `fallback_attempts: 0`, and tokens from the cached `usage` object.

The same first message in a brand-new session (same system prompt, no history) is a cache hit on the second call, even from a different session.

### SQLite Session Storage

SQLite is chosen because: it requires no extra infrastructure, its WAL mode gives good concurrent-read performance, and data lives on a named Docker volume (`app_data` → `/data`) so it survives container restarts.

A module-level `threading.Lock` ensures thread-safe connection handling across uvicorn worker threads. All connections use `PRAGMA journal_mode=WAL` and `timeout=30`.

**503 Atomicity:** `store_exchange()` inserts both the user message and the assistant message in a single transaction. It is only called after a successful LLM response. Failed requests (503, 422) add nothing to the database.

---

## Verification

Copy-paste these after `make up` (replace keys if needed):

```bash
# 1. Health check
curl -s http://localhost:8000/health | python -m json.tool
# Expected: {"status": "ok", "keys_configured": 2, "cache_enabled": true}

# 2. First chat – should be LIVE_INFERENCE
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-verify-1","message":"What is the capital of France?"}' \
  | python -m json.tool
# Expected: resolution_source=LIVE_INFERENCE, key_id=key_1, fallback_attempts=0

# 3. Same message in a NEW session – should be LLM_CACHE
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-verify-2","message":"What is the capital of France?"}' \
  | python -m json.tool
# Expected: resolution_source=LLM_CACHE, key_id=null, reply identical to above

# 4. Failover: key_1=429 → key_2 answers
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-failover","message":"Hello, failover test"}' \
  | python -m json.tool
# Expected: key_id=key_2, fallback_attempts=1

# 5. All keys fail → 503, history unchanged
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-allbad","message":"This should fail"}' \
  | python -m json.tool
# Expected: HTTP 503, {"detail": "all providers unavailable"}

# 6. Confirm history is empty after all-fail
curl -s http://localhost:8000/sessions/s-allbad/messages | python -m json.tool
# Expected: 404 {"detail": "session not found"}  (or 200 with empty messages list)

# 7. History endpoint (after steps 2-3)
curl -s http://localhost:8000/sessions/s-verify-1/messages | python -m json.tool

# 8. 404 for unknown session
curl -s http://localhost:8000/sessions/no-such-session/messages | python -m json.tool
# Expected: 404 {"detail": "session not found"}

# 9. 422 cases
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-val","message":""}' | python -m json.tool
# Expected: 422

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-val","message":"   "}' | python -m json.tool
# Expected: 422

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "not-json" | python -m json.tool
# Expected: 422

# 10. Run tests
make test-local
# or:
cd backend && python -m pytest -v
```

### How to see it in the UI

1. Open http://localhost:3000
2. Each assistant reply shows: source badge (LIVE_INFERENCE / LLM_CACHE), model, latency, input tokens, output tokens.
3. Send the same message in a new chat (click **New Chat**) – the second reply shows LLM_CACHE with lower latency.
4. The input clears after sending.

---

## Setup Used

- **Editor:** Google Antigravity (AI-native IDE)
- **AI tools:** Gemini / Claude via Antigravity agent (automated full-stack build)
- **LLM routing:** LiteLLM (key fallback, disk caching)
- **Backend:** FastAPI + uvicorn + SQLite
- **Frontend:** React + Vite + TypeScript
- **Containerization:** Docker (multi-stage builds), Docker Compose, Makefile

---

## Bonus features

| Feature | Implementation |
|---|---|
| **System prompt field** | `data-testid="system-prompt-input"` text field above chat; value sent as `system_prompt` with every request and persisted per-session in SQLite |
| **New Chat button** | Reloads the page → fresh session ID, empty history |
| **Cache-hit badge color** | `LLM_CACHE` badge is green, `LIVE_INFERENCE` is purple for quick visual distinction |
| **Fallback indicator** | Shows "⚡ N fallbacks" on assistant replies that required key failover |
| **Char counter** | Real-time 0/4000 counter turns red when over limit |
| **Graceful error display** | Network/503/422 errors appear as inline error messages; never crashes or shows blank screen |
| **render.yaml** | One-click Render.com deployment with persistent disk |

---

## Grading criterion checklist

| Criterion | Implementation |
|---|---|
| **Resilience / key failover** | `backend/app/llm.py` – loop over key pool, catches all LiteLLM error types, `fallback_attempts` counter, `X-Simulate-Failure` hook |
| **Caching accuracy** | LiteLLM disk cache, `resolution_source` field, `key_id=null` on hits, same message → LLM_CACHE; different message → miss |
| **Token tracking** | Per-request `usage.prompt_tokens` / `usage.completion_tokens` from LiteLLM response; never cumulative |
| **Multi-turn state** | SQLite `messages` table, last 40 messages loaded per session, sessions fully isolated |
| **Full-stack deployment** | React (Vite/TS) + FastAPI, both Dockerized, `docker-compose.yml`, `render.yaml`, `Makefile` |
| **Code quality / docs** | Typed Python + TypeScript, structured logging (no secrets), pytest suite with mocked LLM, this README |

### Manual steps remaining

1. **Paste real API keys** into `.env` (`LLM_API_KEYS=key1,key2`), or into your host's env var dashboard.
2. **Push to a public GitHub repo** (`git init && git add . && git commit -m "init" && git remote add origin <url> && git push`).
3. **Deploy backend** (Render: connect repo → `render.yaml` auto-detects the backend service → set `LLM_API_KEYS` as a secret env var → note the backend URL).
4. **Deploy frontend** (Render, Vercel, or Netlify: set `VITE_API_BASE_URL=<backend URL>` as a build env var → note the frontend URL; also update `ALLOWED_ORIGINS` on the backend).
5. **Update README Live URLs** section with the real deployed URLs.
6. **Record demo video** (2–3 min): show health check, LIVE_INFERENCE reply, same message → LLM_CACHE (null key_id, lower latency), failover (`key_1=429` → key_2), 503 on `*=429`, multi-turn memory ("My name is Sam" → "What is my name?"), UI with all 5 meta fields.
