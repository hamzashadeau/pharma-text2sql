"""LLM client: ordered models, rate-limit retry, bounded total wait.

Two failure modes decide whether a text-to-SQL service is usable in production,
and neither is about prompt quality:

* **429s.** Providers rate-limit in bursts. Retrying blindly turns one slow
  question into a hung worker, so the wait is bounded in *total seconds*, not in
  attempts: with ``retry-after`` honoured, one attempt can last a minute on its
  own and a plain attempt counter stops meaning anything.
* **A model disappearing.** Model names are retired on a provider's schedule,
  not yours. The client takes an ordered list and walks down it, so losing the
  first model degrades latency instead of taking the service down.

Without an API key the client reports ``available = False`` and the pipeline
uses its deterministic planner instead — the demo and the evaluation suite run
offline, for free, in CI.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import httpx

from .config import settings

logger = logging.getLogger(__name__)

RETRY_BASE_DELAY = 2.0
# A provider can announce a multi-minute retry-after; honouring it literally
# would leave the user's request hanging with no hope.
MAX_RETRY_AFTER = 20.0


class LLMUnavailable(RuntimeError):
    """No model could answer — the caller falls back or surfaces the failure."""


@dataclass
class Completion:
    text: str
    model: str
    elapsed_s: float


def available() -> bool:
    return settings.llm_available


def complete(system: str, user: str, *, json_mode: bool = False) -> Completion:
    """Ask the first model that answers. Raises :class:`LLMUnavailable`."""
    if not settings.llm_available:
        raise LLMUnavailable("no API key configured")

    started = time.monotonic()
    waited = 0.0
    last_error = "no model attempted"

    for model in settings.models:
        for attempt in range(settings.max_retries):
            try:
                text = _post(model, system, user, json_mode=json_mode)
                return Completion(text=text, model=model, elapsed_s=time.monotonic() - started)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                last_error = f"{model}: HTTP {status}"
                if status == 429:
                    delay = _retry_delay(exc.response, attempt)
                    if waited + delay > settings.max_total_wait_s:
                        logger.warning("wait budget exhausted on %s, trying next model", model)
                        break
                    logger.info("rate limited by %s, waiting %.1fs", model, delay)
                    time.sleep(delay)
                    waited += delay
                    continue
                if 500 <= status < 600:
                    time.sleep(RETRY_BASE_DELAY)
                    waited += RETRY_BASE_DELAY
                    continue
                break  # 4xx other than 429: the next attempt fails identically
            except httpx.HTTPError as exc:
                last_error = f"{model}: {exc}"
                break

    raise LLMUnavailable(last_error)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_RETRY_AFTER)
        except ValueError:
            pass
    return min(RETRY_BASE_DELAY * (2 ** attempt), MAX_RETRY_AFTER)


def _post(model: str, system: str, user: str, *, json_mode: bool) -> str:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=settings.request_timeout_s) as client:
        response = client.post(
            f"{settings.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()

    return body["choices"][0]["message"]["content"]


def complete_json(system: str, user: str) -> dict:
    """Completion parsed as JSON, tolerant of a fenced code block."""
    raw = complete(system, user, json_mode=True).text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(raw)
