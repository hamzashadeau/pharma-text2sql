"""Configuration, read from the environment with a ``P2S_`` prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"P2S_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- data -------------------------------------------------------------
    dataset_path: Path = field(
        default_factory=lambda: Path(_env("DATASET", str(ROOT / "data" / "sales.parquet")))
    )
    table_name: str = "sales"

    # --- LLM --------------------------------------------------------------
    # Ordered list: the second model answers when the first is rate-limited.
    models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip()
            for m in _env("MODELS", "mistralai/mistral-small,mistralai/mistral-large").split(",")
            if m.strip()
        )
    )
    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    api_base: str = field(
        default_factory=lambda: _env("API_BASE", "https://openrouter.ai/api/v1")
    )
    request_timeout_s: int = field(default_factory=lambda: _env_int("TIMEOUT", 60))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    max_total_wait_s: int = field(default_factory=lambda: _env_int("MAX_WAIT", 45))

    # --- guardrails -------------------------------------------------------
    # Every generated query is capped, whatever the model asked for.
    max_rows: int = field(default_factory=lambda: _env_int("MAX_ROWS", 500))
    sandbox_timeout_s: int = field(default_factory=lambda: _env_int("SANDBOX_TIMEOUT", 15))
    sandbox_memory_mb: int = field(default_factory=lambda: _env_int("SANDBOX_MEMORY_MB", 512))

    @property
    def llm_available(self) -> bool:
        """True when a real model can be called.

        Without a key the pipeline falls back to the deterministic planner, so
        the demo and the evaluation suite still run — offline and for free.
        """
        return bool(self.api_key)


settings = Settings()
