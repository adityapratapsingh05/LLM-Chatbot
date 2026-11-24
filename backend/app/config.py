"""
Backend configuration – all values come from environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pathlib import Path

# Load .env file at module import time
# Look for .env in backend/ directory (parent of app/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_API_KEYS: List[str]
    LLM_MODEL: str
    DEFAULT_SYSTEM_PROMPT: str
    CACHE_DIR: str

    # ── Storage ───────────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── Network ───────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str]

    # ── Feature flags ─────────────────────────────────────────────────────────
    ENABLE_FAILURE_SIMULATION: bool

    def __init__(self) -> None:
        raw_keys = os.environ.get("LLM_API_KEYS", "")
        self.LLM_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

        self.LLM_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-2.0-flash")
        self.DEFAULT_SYSTEM_PROMPT = os.environ.get(
            "DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant."
        )
        self.CACHE_DIR = os.environ.get("CACHE_DIR", ":memory:")
        self.DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///:memory:")

        raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
        self.ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]

        self.ENABLE_FAILURE_SIMULATION = (
            os.environ.get("ENABLE_FAILURE_SIMULATION", "false").lower() == "true"
        )

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def db_path(self) -> str:
        """Return the filesystem path from DATABASE_URL."""
        url = self.DATABASE_URL
        # Handle in-memory database
        if url == "sqlite:///:memory:":
            return ":memory:"
        # sqlite:////abs/path  -> /abs/path
        # sqlite:///rel/path   -> rel/path
        if url.startswith("sqlite:////"):
            return url[len("sqlite:///"):]
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///"):]
        return url

    @property
    def key_pool(self) -> List[tuple[str, str]]:
        """Return [(key_id, api_key), ...]."""
        return [(f"key_{i+1}", k) for i, k in enumerate(self.LLM_API_KEYS)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
