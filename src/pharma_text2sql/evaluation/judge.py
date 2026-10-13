"""LLM-as-judge, for the part arithmetic cannot settle.

:mod:`faithfulness` answers "are these figures in the table" completely and for
free. It cannot answer "is this the right table for the question" — a query
that returns 2023 when the user asked for 2024 is perfectly faithful to its own
result. That judgement needs a reader, and when no labelled answer exists a
model is the only reader available at this scale.

Three rules keep it from becoming theatre:

* **It is never silently skipped.** With no API key configured, every verdict
  is ``unavailable`` and the report says so. A judge that quietly returns
  "pass" when it did not run is worse than no judge.
* **It judges against something.** The question, the SQL, the result preview
  and — when the question set has one — the stated target. A judge given only
  an answer rates fluency.
* **It is asked to fail.** The rubric defaults to ``fail`` when the evidence is
  insufficient, because a judge that wants to be agreeable will agree.

The verdicts are a signal for a human to review, never a gate. Nothing in this
module should be wired to a build that blocks merges.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from .. import llm

logger = logging.getLogger(__name__)

PASS, FAIL, UNAVAILABLE = "pass", "fail", "unavailable"

SYSTEM_PROMPT = """You check whether an analytics answer addresses the question asked.

You receive the question, the SQL that ran, a preview of the result, and the
answer given to the user. Judge only whether the ANSWER AND QUERY ADDRESS THE
QUESTION — not whether the writing is good, and not whether the figures are
arithmetically right (that is checked separately).

Fail it when:
- the query filters or groups by something the question did not ask for, or
  misses a filter the question did ask for;
- the answer states something the result does not show;
- the answer dodges the question.

Default to "fail" when the evidence is insufficient to tell. Being unsure is a
reason to flag for review, not to approve.

Return ONLY JSON: {"verdict": "pass" | "fail", "reason": "<one sentence>"}"""


@dataclass
class Verdict:
    verdict: str
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.verdict != UNAVAILABLE

    @property
    def passed(self) -> bool:
        return self.verdict == PASS


def available() -> bool:
    return llm.available()


def judge(
    question: str,
    answer: str,
    sql: str = "",
    frame: pd.DataFrame | None = None,
    target: str = "",
    preview_rows: int = 10,
) -> Verdict:
    """Judge one answer. Never raises — a broken judge must not fail a run."""
    if not llm.available():
        return Verdict(UNAVAILABLE, "no API key configured")

    preview = ""
    if frame is not None and not frame.empty:
        try:
            preview = frame.head(preview_rows).to_markdown(index=False)
        except Exception:
            preview = frame.head(preview_rows).to_string(index=False)

    user = "\n".join(filter(None, [
        f"Question: {question}",
        f"Stated target: {target}" if target else "",
        f"SQL:\n{sql}" if sql else "",
        f"Result ({0 if frame is None else len(frame)} rows):\n{preview}" if preview else
        "Result: empty",
        f"Answer given to the user: {answer}",
    ]))

    try:
        raw = llm.complete_json(SYSTEM_PROMPT, user)
    except Exception as exc:
        logger.warning("judge call failed: %s", exc)
        return Verdict(UNAVAILABLE, f"judge call failed: {exc}")

    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in (PASS, FAIL):
        return Verdict(UNAVAILABLE, f"unparseable verdict: {raw!r}")

    return Verdict(verdict, str(raw.get("reason", "")).strip())


def judge_all(items: list[dict], preview_rows: int = 10) -> list[Verdict]:
    """Judge a batch. Each item carries the keys :func:`judge` accepts."""
    if not llm.available():
        return [Verdict(UNAVAILABLE, "no API key configured") for _ in items]
    return [judge(preview_rows=preview_rows, **item) for item in items]


def summarise(verdicts: list[Verdict]) -> str:
    if not verdicts:
        return "judge: nothing to score"

    unavailable = sum(1 for v in verdicts if not v.available)
    if unavailable == len(verdicts):
        return f"judge: not run ({verdicts[0].reason})"

    scored = [v for v in verdicts if v.available]
    passed = sum(1 for v in scored if v.passed)
    line = f"judge: {passed}/{len(scored)} pass ({passed / len(scored):.0%})"
    if unavailable:
        line += f", {unavailable} not scored"
    return line
