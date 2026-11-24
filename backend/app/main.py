"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.config import get_settings
from app.database import (
    get_all_messages,
    get_or_create_session,
    get_session_history,
    init_db,
    session_exists,
    store_exchange,
    update_session_system_prompt,
)
from app.llm import call_llm, init_cache

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────

_startup_ok = False
_cache_enabled = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_ok, _cache_enabled
    settings = get_settings()

    # Validate key pool
    if len(settings.LLM_API_KEYS) < 2:
        logger.error(
            "LLM_API_KEYS must contain at least 2 keys (got %d). "
            "The /health endpoint will report failure.",
            len(settings.LLM_API_KEYS),
        )
        # We still start (so the container doesn't crash) but health will be non-ok.
    else:
        _startup_ok = True

    # Init DB
    try:
        init_db()
        logger.info("Database initialised at %s", settings.db_path)
    except Exception as exc:
        logger.warning("DB init warning: %s", exc)
        # Continue anyway - tables will be created on first use

    # Init cache
    _cache_enabled = init_cache()

    logger.info(
        "App ready | model=%s | keys=%d | cache=%s | sim=%s",
        settings.LLM_MODEL,
        len(settings.LLM_API_KEYS),
        _cache_enabled,
        settings.ENABLE_FAILURE_SIMULATION,
    )
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="LLM Chatbot", lifespan=lifespan)

# ── CORS (must be added before routes) ───────────────────────────────────────
def _build_cors():
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[],
    )

_build_cors()


# ── CORS ──────────────────────────────────────────────────────────────────────




# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Pydantic v2 errors may contain non-serializable objects in 'ctx'; sanitize them.
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items() if k != "ctx" or isinstance(v, dict)}
        if isinstance(obj, list):
            return [_sanitize(i) for i in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    errors = _sanitize(exc.errors())
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("session_id must not be empty")
        if len(v) > 128:
            raise ValueError("session_id must be <= 128 characters")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("message must not be empty or whitespace-only")
        if len(trimmed) > 4000:
            raise ValueError("message must be <= 4000 characters")
        return trimmed  # return trimmed value


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    settings = get_settings()
    # Check DB reachable
    db_ok = False
    try:
        from app.database import get_conn
        with get_conn() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)

    keys_configured = len(settings.LLM_API_KEYS)
    ok = _startup_ok and db_ok and keys_configured >= 2

    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "keys_configured": keys_configured,
                "cache_enabled": _cache_enabled,
            },
        )

    return {
        "status": "ok",
        "keys_configured": keys_configured,
        "cache_enabled": _cache_enabled,
    }


@app.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
):
    t_start = time.perf_counter()
    settings = get_settings()

    # Resolve / create session and system prompt
    effective_system_prompt = settings.DEFAULT_SYSTEM_PROMPT

    # If session exists and caller doesn't supply system_prompt, use stored one.
    if session_exists(req.session_id):
        # Load existing prompt unless overridden
        stored_prompt = get_or_create_session(req.session_id, effective_system_prompt)
        if req.system_prompt is not None:
            # Caller overrides – update stored
            effective_system_prompt = req.system_prompt
            update_session_system_prompt(req.session_id, effective_system_prompt)
        else:
            effective_system_prompt = stored_prompt
    else:
        # New session
        if req.system_prompt is not None:
            effective_system_prompt = req.system_prompt
        get_or_create_session(req.session_id, effective_system_prompt)

    # Build message list for LLM
    history = get_session_history(req.session_id, limit=40)
    llm_messages = (
        [{"role": "system", "content": effective_system_prompt}]
        + [{"role": m["role"], "content": m["content"]} for m in history]
        + [{"role": "user", "content": req.message}]
    )

    # Call LLM (with failover / cache)
    try:
        reply, metadata = await call_llm(
            messages=llm_messages,
            simulate_header=x_simulate_failure,
            enable_simulation=settings.ENABLE_FAILURE_SIMULATION,
        )
    except RuntimeError as exc:
        if "all providers unavailable" in str(exc):
            raise HTTPException(status_code=503, detail="all providers unavailable")
        raise

    # Measure total latency (overwrite the per-call latency for cache hits too)
    total_latency_ms = int((time.perf_counter() - t_start) * 1000)
    metadata["latency_ms"] = total_latency_ms

    # Store exchange (only on success)
    store_meta = {
        "resolution_source": metadata["resolution_source"],
        "input_tokens": metadata["input_tokens"],
        "output_tokens": metadata["output_tokens"],
    }
    store_exchange(req.session_id, req.message, reply, store_meta)

    return {
        "session_id": req.session_id,
        "reply": reply,
        "metadata": metadata,
    }


@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    messages = get_all_messages(session_id)
    return {"session_id": session_id, "messages": messages}
