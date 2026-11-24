"""
LiteLLM integration: key-pool failover, disk caching, failure simulation hook.

Cache strategy:
  - LiteLLM 1.110+ dropped the `type="disk"` cache option.  We inject a custom
    BaseCache subclass backed by `diskcache` as the `_backend` to `Cache(...)`.
    This keeps LiteLLM's native caching pipeline (key derivation, callbacks,
    cache_hit detection) while storing data persistently on disk.
  - Cache key = model + full message list (system prompt + history + new message).
    The api_key is NOT part of the key so a hit for key_1 also serves key_2.
  - Temperature is fixed at 0 for deterministic, cacheable responses.
  - max_messages is set to None so long conversations are still cached.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import diskcache
import litellm
from litellm import Cache
from litellm.caching.base_cache import BaseCache
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Disk-backed BaseCache implementation ──────────────────────────────────────

class _DiskCache(BaseCache):
    """
    diskcache-backed BaseCache that LiteLLM injects via Cache(_backend=...).
    Keys and values are stored as JSON on disk so they survive container restarts.
    """

    def __init__(self, cache_dir: str, ttl: int = 86400 * 30) -> None:  # 30-day TTL
        super().__init__(default_ttl=ttl)
        os.makedirs(cache_dir, exist_ok=True)
        self._dc: diskcache.Cache = diskcache.Cache(cache_dir)
        self._ttl = ttl

    # -- sync -----------------------------------------------------------------

    def set_cache(self, key: str, value: Any, **kwargs) -> None:
        try:
            ttl = kwargs.get("ttl") or self._ttl
            self._dc.set(key, json.dumps(value, default=str), expire=ttl)
        except Exception as exc:
            logger.debug("DiskCache.set_cache error (ignored): %s", exc)

    def get_cache(self, key: str, **kwargs) -> Any:
        try:
            raw = self._dc.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("DiskCache.get_cache error (ignored): %s", exc)
            return None

    # -- async (delegated to sync) --------------------------------------------

    async def async_set_cache(self, key: str, value: Any, **kwargs) -> None:
        self.set_cache(key, value, **kwargs)

    async def async_get_cache(self, key: str, **kwargs) -> Any:
        return self.get_cache(key, **kwargs)

    async def async_set_cache_pipeline(self, cache_list: list, **kwargs) -> None:
        for k, v in cache_list:
            self.set_cache(k, v, **kwargs)

    async def batch_cache_write(self, key: str, value: Any, **kwargs) -> None:
        self.set_cache(key, value, **kwargs)

    async def disconnect(self) -> None:
        try:
            self._dc.close()
        except Exception:
            pass

    async def test_connection(self) -> dict:
        try:
            self._dc.set("__ping__", "1", expire=1)
            return {"status": "success", "message": "disk cache ok"}
        except Exception as exc:
            return {"status": "failed", "message": str(exc), "error": str(exc)}


# ── Cache init (called once at startup) ───────────────────────────────────────

def init_cache() -> bool:
    settings = get_settings()
    cache_dir = settings.CACHE_DIR
    try:
        backend = _DiskCache(cache_dir=cache_dir)
        litellm.cache = Cache(
            _backend=backend,
            max_messages=None,  # don't skip long conversations
        )
        litellm.enable_cache()
        logger.info("LiteLLM disk cache (diskcache) enabled at %s", cache_dir)
        return True
    except Exception as exc:
        logger.warning("Disk cache init failed (%s); falling back to in-memory cache", exc)
        try:
            litellm.cache = Cache(type="local", max_messages=None)
            litellm.enable_cache()
            return True
        except Exception:
            return False


# ── Failure simulation ────────────────────────────────────────────────────────

def _parse_simulate_header(header: str) -> Dict[str, int]:
    """Parse 'key_1=429,key_2=500' or '*=429' into {key_id: status_code}."""
    result: Dict[str, int] = {}
    for part in header.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        kid, _, code_str = part.partition("=")
        kid = kid.strip()
        code_str = code_str.strip()
        if not kid or not code_str:
            continue
        try:
            code = int(code_str)
        except ValueError:
            continue
        if code not in (429, 500, 502, 503):
            continue
        result[kid] = code
    return result


def _make_litellm_exception(status_code: int, key_id: str) -> Exception:
    msg = f"[simulated] {status_code} for {key_id}"
    if status_code == 429:
        return RateLimitError(message=msg, llm_provider="gemini", model="gemini")
    if status_code == 503:
        return ServiceUnavailableError(message=msg, llm_provider="gemini", model="gemini")
    return InternalServerError(message=msg, llm_provider="gemini", model="gemini")


# ── Retriable exception predicate ─────────────────────────────────────────────

_RETRYABLE = (
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
    APIConnectionError,
    Timeout,
    AuthenticationError,
    BadRequestError,
)


# ── Main inference function ───────────────────────────────────────────────────

async def call_llm(
    messages: List[Dict[str, str]],
    simulate_header: Optional[str],
    enable_simulation: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Try each key in order; return (reply_text, metadata_dict).
    Raises RuntimeError('all providers unavailable') if all keys fail.
    """
    settings = get_settings()
    model = settings.LLM_MODEL
    key_pool = settings.key_pool  # [(key_id, api_key), ...]

    # Parse failure simulation
    sim_map: Dict[str, int] = {}
    bypass_cache = False
    if enable_simulation and simulate_header:
        sim_map = _parse_simulate_header(simulate_header)
        if sim_map:
            bypass_cache = True  # don't serve cached reply during failover test

    # ── Cache lookup (skip when bypass_cache) ─────────────────────────────────
    if not bypass_cache:
        cached = await _try_cache(model, messages)
        if cached is not None:
            return cached

    # ── Live call with failover ───────────────────────────────────────────────
    fallback_attempts = 0
    last_exc: Optional[Exception] = None

    for key_id, api_key in key_pool:
        # Check simulation
        sim_code = sim_map.get(key_id) or sim_map.get("*")
        if sim_code is not None:
            last_exc = _make_litellm_exception(sim_code, key_id)
            logger.info("Simulating %s failure for %s", sim_code, key_id)
            fallback_attempts += 1
            continue

        try:
            t0 = time.perf_counter()
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                api_key=api_key,
                caching=True,
                temperature=0,
                timeout=30,
                num_retries=0,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            reply = response.choices[0].message.content or ""
            usage = response.usage or {}

            cache_hit = getattr(response, "_hidden_params", {}).get("cache_hit", False)

            # Determine token counts
            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
            else:
                input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

            metadata: Dict[str, Any] = {
                "resolution_source": "LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
                "model": model,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "key_id": None if cache_hit else key_id,
                "fallback_attempts": 0 if cache_hit else fallback_attempts,
            }
            return reply, metadata

        except _RETRYABLE as exc:
            logger.warning("Key %s failed (%s): %s", key_id, type(exc).__name__, str(exc)[:120])
            last_exc = exc
            fallback_attempts += 1
            continue
        except Exception as exc:
            logger.error("Unexpected error from key %s: %s", key_id, exc)
            last_exc = exc
            fallback_attempts += 1
            continue

    raise RuntimeError("all providers unavailable") from last_exc


# ── Cache probe ───────────────────────────────────────────────────────────────

async def _try_cache(model: str, messages: List[Dict[str, str]]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (reply, metadata) from cache, or None on miss."""
    if litellm.cache is None:
        return None
    try:
        # LiteLLM cache.get_cache expects the same args as completion
        cached_resp = litellm.cache.get_cache(
            model=model,
            messages=messages,
            temperature=0,
        )
        if cached_resp is None:
            return None

        # cached_resp may be a dict (deserialized from disk) or a ModelResponse object
        if isinstance(cached_resp, dict):
            choices = cached_resp.get("choices", [])
            if not choices:
                return None
            choice = choices[0]
            if isinstance(choice, dict):
                reply = choice.get("message", {}).get("content", "") or ""
            else:
                reply = getattr(getattr(choice, "message", None), "content", "") or ""
            usage_raw = cached_resp.get("usage", {})
            if isinstance(usage_raw, dict):
                input_tokens = usage_raw.get("prompt_tokens", 0) or 0
                output_tokens = usage_raw.get("completion_tokens", 0) or 0
            else:
                input_tokens = getattr(usage_raw, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage_raw, "completion_tokens", 0) or 0
        else:
            reply = cached_resp.choices[0].message.content or ""
            usage = cached_resp.usage or {}
            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
            else:
                input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

        if not reply:
            return None

        metadata: Dict[str, Any] = {
            "resolution_source": "LLM_CACHE",
            "model": model,
            "latency_ms": 0,  # will be overwritten by caller
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "key_id": None,
            "fallback_attempts": 0,
        }
        return reply, metadata
    except Exception as exc:
        logger.debug("Cache probe error (ignored): %s", exc)
        return None
