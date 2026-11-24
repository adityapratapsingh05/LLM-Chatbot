"""
Backend test suite – LLM calls are fully mocked (monkeypatched).
Run with: pytest -v
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# ── Fixtures & Setup ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Each test gets a fresh DB and env vars."""
    db_file = str(tmp_path / "test.db")
    cache_dir = str(tmp_path / "cache")
    monkeypatch.setenv("LLM_API_KEYS", "fake_key_1,fake_key_2")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("CACHE_DIR", cache_dir)
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    # Reset lru_cache on settings
    from app.config import get_settings
    get_settings.cache_clear()
    # Reset DB path cache
    import app.database as db_module
    db_module._db_path = None
    yield


@pytest.fixture
def client(isolated_env):
    """Fresh TestClient with clean app state."""
    # Re-import with fresh settings
    from app.config import get_settings
    get_settings.cache_clear()
    import app.database as db_module
    db_module._db_path = None

    # Mock litellm cache init so it doesn't fail in tests
    with patch("app.llm.init_cache", return_value=True), \
         patch("app.main._cache_enabled", True), \
         patch("app.main._startup_ok", True):
        from app.main import app
        # Init DB
        from app.database import init_db
        init_db()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _make_llm_response(
    content: str = "Paris is the capital of France.",
    prompt_tokens: int = 42,
    completion_tokens: int = 9,
    cache_hit: bool = False,
) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response._hidden_params = {"cache_hit": cache_hit}
    return response


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["keys_configured"] == 2
        assert "cache_enabled" in body

    def test_health_db_reachable(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


# ── 422 Validation ────────────────────────────────────────────────────────────

class Test422:
    def test_missing_message(self, client):
        resp = client.post("/chat", json={"session_id": "s-test"})
        assert resp.status_code == 422
        assert "detail" in resp.json()

    def test_empty_message(self, client):
        resp = client.post("/chat", json={"session_id": "s-test", "message": ""})
        assert resp.status_code == 422

    def test_whitespace_only_message(self, client):
        resp = client.post("/chat", json={"session_id": "s-test", "message": "   "})
        assert resp.status_code == 422

    def test_message_too_long(self, client):
        resp = client.post("/chat", json={"session_id": "s-test", "message": "x" * 4001})
        assert resp.status_code == 422

    def test_missing_session_id(self, client):
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 422

    def test_empty_session_id(self, client):
        resp = client.post("/chat", json={"session_id": "", "message": "hello"})
        assert resp.status_code == 422

    def test_malformed_json(self, client):
        resp = client.post(
            "/chat",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert "detail" in resp.json()

    def test_no_stack_trace_in_error(self, client):
        resp = client.post("/chat", json={"session_id": "s", "message": ""})
        body = resp.text
        assert "traceback" not in body.lower()
        assert "file " not in body.lower()


# ── Successful chat ───────────────────────────────────────────────────────────

class TestChatSuccess:
    def test_basic_chat_metadata_shape(self, client):
        mock_resp = _make_llm_response()
        with patch("app.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post("/chat", json={"session_id": "s-1", "message": "What is the capital of France?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "s-1"
        assert body["reply"] == "Paris is the capital of France."
        m = body["metadata"]
        assert m["resolution_source"] == "LIVE_INFERENCE"
        assert m["model"] == "gemini/gemini-2.0-flash"
        assert isinstance(m["latency_ms"], int)
        assert m["input_tokens"] == 42
        assert m["output_tokens"] == 9
        assert m["key_id"] == "key_1"
        assert m["fallback_attempts"] == 0

    def test_trimmed_message_used(self, client):
        """Message is trimmed before sending to LLM."""
        mock_resp = _make_llm_response()
        captured = []

        async def fake_completion(**kwargs):
            captured.append(kwargs["messages"])
            return mock_resp

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post("/chat", json={"session_id": "s-trim", "message": "  hello  "})
        assert resp.status_code == 200
        # Last message should be trimmed
        assert captured[0][-1]["content"] == "hello"

    def test_per_request_tokens_not_cumulative(self, client):
        """Two first messages in separate sessions should report same input_tokens."""
        mock_resp = _make_llm_response(prompt_tokens=42)
        with patch("app.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                r1 = client.post("/chat", json={"session_id": "s-tok1", "message": "Hello world"})
                r2 = client.post("/chat", json={"session_id": "s-tok2", "message": "Hello world"})
        assert r1.json()["metadata"]["input_tokens"] == r2.json()["metadata"]["input_tokens"]


# ── Multi-turn & session isolation ───────────────────────────────────────────

class TestMultiTurn:
    def test_session_remembers_history(self, client):
        """Bot should use history from earlier turns in same session."""
        call_count = [0]

        async def fake_completion(**kwargs):
            call_count[0] += 1
            msgs = kwargs["messages"]
            if call_count[0] == 1:
                return _make_llm_response("My name is Sam, got it!")
            else:
                # Second call: history should include first exchange
                roles = [m["role"] for m in msgs]
                assert "user" in roles
                assert "assistant" in roles
                return _make_llm_response("Your name is Sam.")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                r1 = client.post("/chat", json={"session_id": "s-mem", "message": "My name is Sam"})
                r2 = client.post("/chat", json={"session_id": "s-mem", "message": "What is my name?"})
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_sessions_are_isolated(self, client):
        """Two sessions should not share history."""
        histories: Dict[str, list] = {}

        async def fake_completion(**kwargs):
            msgs = kwargs["messages"]
            sid = "unknown"
            for m in msgs:
                if m["role"] == "user":
                    sid = m["content"]
            histories[sid] = msgs
            return _make_llm_response("ok")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                client.post("/chat", json={"session_id": "s-A", "message": "Session A message"})
                client.post("/chat", json={"session_id": "s-B", "message": "Session B message"})
                # Second turn in A should not see B's history
                client.post("/chat", json={"session_id": "s-A", "message": "turn 2 of A"})

        # Check DB isolation
        from app.database import get_all_messages
        msgs_a = get_all_messages("s-A")
        msgs_b = get_all_messages("s-B")
        assert all("Session B" not in m["content"] for m in msgs_a)
        assert all("Session A" not in m["content"] for m in msgs_b)


# ── History endpoint ──────────────────────────────────────────────────────────

class TestHistory:
    def test_history_order_oldest_first(self, client):
        mock_resp = _make_llm_response()
        with patch("app.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                client.post("/chat", json={"session_id": "s-hist", "message": "First"})
                client.post("/chat", json={"session_id": "s-hist", "message": "Second"})

        resp = client.get("/sessions/s-hist/messages")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "s-hist"
        msgs = body["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "First"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == "Second"

    def test_history_404_unknown_session(self, client):
        resp = client.get("/sessions/nonexistent-session-xyz/messages")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "session not found"

    def test_assistant_message_has_metadata(self, client):
        mock_resp = _make_llm_response()
        with patch("app.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                client.post("/chat", json={"session_id": "s-meta", "message": "hi"})

        resp = client.get("/sessions/s-meta/messages")
        msgs = resp.json()["messages"]
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "metadata" in assistant_msg
        m = assistant_msg["metadata"]
        assert "resolution_source" in m
        assert "input_tokens" in m
        assert "output_tokens" in m


# ── Cache ────────────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_hit_returns_llm_cache_source(self, client):
        cached_meta = {
            "resolution_source": "LLM_CACHE",
            "model": "gemini/gemini-2.0-flash",
            "latency_ms": 0,
            "input_tokens": 42,
            "output_tokens": 9,
            "key_id": None,
            "fallback_attempts": 0,
        }
        with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=("Paris is the capital.", cached_meta)):
            resp = client.post("/chat", json={"session_id": "s-cache", "message": "What is the capital of France?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["resolution_source"] == "LLM_CACHE"
        assert body["metadata"]["key_id"] is None
        assert body["metadata"]["fallback_attempts"] == 0
        assert body["reply"] == "Paris is the capital."

    def test_cache_hit_null_key_id(self, client):
        cached_meta = {
            "resolution_source": "LLM_CACHE",
            "model": "gemini/gemini-2.0-flash",
            "latency_ms": 0,
            "input_tokens": 10,
            "output_tokens": 5,
            "key_id": None,
            "fallback_attempts": 0,
        }
        with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=("ok", cached_meta)):
            resp = client.post("/chat", json={"session_id": "s-ck", "message": "test"})
        assert resp.json()["metadata"]["key_id"] is None

    def test_different_message_not_cached(self, client):
        """Different message should NOT trigger a cache return (miss)."""
        mock_resp = _make_llm_response("Different answer")
        with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
            with patch("app.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
                resp = client.post("/chat", json={"session_id": "s-diff", "message": "Completely different query"})
        assert resp.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    def test_cache_hit_faster_latency(self, client):
        """Cache hits should be measurably lower latency (we just check it's set)."""
        import asyncio

        async def slow_cache(*args, **kwargs):
            cached_meta = {
                "resolution_source": "LLM_CACHE",
                "model": "gemini/gemini-2.0-flash",
                "latency_ms": 0,
                "input_tokens": 10,
                "output_tokens": 5,
                "key_id": None,
                "fallback_attempts": 0,
            }
            return "cached reply", cached_meta

        with patch("app.llm._try_cache", side_effect=slow_cache):
            resp = client.post("/chat", json={"session_id": "s-lat", "message": "latency test"})

        assert resp.status_code == 200
        assert isinstance(resp.json()["metadata"]["latency_ms"], int)


# ── Failover ─────────────────────────────────────────────────────────────────

class TestFailover:
    def test_key1_429_falls_back_to_key2(self, client):
        from litellm.exceptions import RateLimitError

        call_log = []

        async def fake_completion(**kwargs):
            api_key = kwargs.get("api_key", "")
            call_log.append(api_key)
            if api_key == "fake_key_1":
                raise RateLimitError(message="rate limit", llm_provider="gemini", model="gemini")
            return _make_llm_response("Answer from key 2")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post(
                    "/chat",
                    json={"session_id": "s-fo1", "message": "hello"},
                    headers={"X-Simulate-Failure": "key_1=429"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["key_id"] == "key_2"
        assert body["metadata"]["fallback_attempts"] == 1

    def test_key1_500_falls_back_to_key2(self, client):
        from litellm.exceptions import InternalServerError

        async def fake_completion(**kwargs):
            api_key = kwargs.get("api_key", "")
            if api_key == "fake_key_1":
                raise InternalServerError(message="server error", llm_provider="gemini", model="gemini")
            return _make_llm_response("key2 answer")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post(
                    "/chat",
                    json={"session_id": "s-fo2", "message": "hello"},
                    headers={"X-Simulate-Failure": "key_1=500"},
                )

        assert resp.status_code == 200
        assert resp.json()["metadata"]["key_id"] == "key_2"

    def test_all_keys_fail_returns_503(self, client):
        from litellm.exceptions import RateLimitError

        async def fake_completion(**kwargs):
            raise RateLimitError(message="rate limit", llm_provider="gemini", model="gemini")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post(
                    "/chat",
                    json={"session_id": "s-fo3", "message": "hello"},
                    headers={"X-Simulate-Failure": "*=429"},
                )

        assert resp.status_code == 503
        assert resp.json()["detail"] == "all providers unavailable"

    def test_all_keys_fail_nothing_stored(self, client):
        """On 503, nothing should be added to history."""
        from litellm.exceptions import RateLimitError

        async def fake_completion(**kwargs):
            raise RateLimitError(message="rate limit", llm_provider="gemini", model="gemini")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post(
                    "/chat",
                    json={"session_id": "s-fo4", "message": "should not be stored"},
                    headers={"X-Simulate-Failure": "*=429"},
                )

        assert resp.status_code == 503
        # Session may not exist at all, or exist with 0 messages
        from app.database import session_exists, get_all_messages
        if session_exists("s-fo4"):
            msgs = get_all_messages("s-fo4")
            assert len(msgs) == 0

    def test_key1_429_key2_500_all_fail_503(self, client):
        from litellm.exceptions import InternalServerError, RateLimitError

        async def fake_completion(**kwargs):
            api_key = kwargs.get("api_key", "")
            if api_key == "fake_key_1":
                raise RateLimitError(message="rl", llm_provider="gemini", model="gemini")
            raise InternalServerError(message="ise", llm_provider="gemini", model="gemini")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post(
                    "/chat",
                    json={"session_id": "s-fo5", "message": "fail both"},
                    headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
                )

        assert resp.status_code == 503
        assert resp.json()["detail"] == "all providers unavailable"

    def test_star_429_all_fail_nothing_in_history(self, client):
        from litellm.exceptions import RateLimitError

        async def fake_completion(**kwargs):
            raise RateLimitError(message="rl", llm_provider="gemini", model="gemini")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                r = client.post(
                    "/chat",
                    json={"session_id": "s-star", "message": "star fail"},
                    headers={"X-Simulate-Failure": "*=429"},
                )
        assert r.status_code == 503

        resp = client.get("/sessions/s-star/messages")
        # Either 404 (session never created) or empty messages
        assert resp.status_code in (404, 200)
        if resp.status_code == 200:
            assert len(resp.json()["messages"]) == 0

    def test_simulation_ignored_when_disabled(self, client, monkeypatch):
        """X-Simulate-Failure header must be ignored when ENABLE_FAILURE_SIMULATION=false."""
        monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "false")
        from app.config import get_settings
        get_settings.cache_clear()

        mock_resp = _make_llm_response("normal answer")
        with patch("app.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                resp = client.post(
                    "/chat",
                    json={"session_id": "s-nodis", "message": "hi"},
                    headers={"X-Simulate-Failure": "key_1=429"},
                )

        # Should succeed normally (key_1 used, no failover)
        assert resp.status_code == 200
        assert resp.json()["metadata"]["key_id"] == "key_1"
        assert resp.json()["metadata"]["fallback_attempts"] == 0


# ── System prompt ─────────────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_custom_system_prompt_persisted(self, client):
        captured = []

        async def fake_completion(**kwargs):
            captured.append(kwargs["messages"])
            return _make_llm_response("ok")

        with patch("app.llm.litellm.acompletion", side_effect=fake_completion):
            with patch("app.llm._try_cache", new_callable=AsyncMock, return_value=None):
                # First request with custom system prompt
                client.post("/chat", json={
                    "session_id": "s-sys",
                    "message": "hi",
                    "system_prompt": "You are a pirate.",
                })
                # Second request without system_prompt should use the stored one
                client.post("/chat", json={"session_id": "s-sys", "message": "hello again"})

        # Both calls should have the same system prompt
        assert captured[0][0]["content"] == "You are a pirate."
        assert captured[1][0]["content"] == "You are a pirate."
