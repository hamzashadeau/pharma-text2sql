"""The pipeline, in two phases on purpose.

``explain()`` understands the question and stops. ``execute()`` runs it. The
split is the whole point: between the two, a human sees what the system
believes it was asked and either confirms it or corrects it.

That gate is cheap to build and it is what makes the system defensible where it
matters. A text-to-SQL answer is confidently worded whether or not the query
meant what the user intended; without a confirmation step, a misread question
becomes a number in a slide deck with nothing marking it as wrong. In a
regulated setting the same gate is also the audit trail: what was asked, how it
was understood, what was approved, what ran.

    explain  ->  extract -> resolve -> interpret            (no SQL runs)
    execute  ->  extract -> resolve -> generate -> guard
                 -> run -> narrate -> suggest a chart
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import entities, llm, planner, resolver, schema, sql_guard

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    question: str
    interpretation: str
    resolved: dict[str, list[str]] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification: str = ""
    extraction: entities.Extraction | None = None


@dataclass
class Answer:
    question: str
    ok: bool
    sql: str = ""
    narration: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    chart: dict[str, str] | None = None
    error: str = ""
    elapsed_s: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


# ---------------------------------------------------------------------------
# Phase 1 — understand, do not execute
# ---------------------------------------------------------------------------

def explain(question: str) -> Explanation:
    extraction = entities.extract(question)
    resolution = resolver.resolve(extraction.entities)

    return Explanation(
        question=question,
        interpretation=_interpretation(extraction, resolution),
        resolved=dict(resolution.resolved),
        needs_clarification=resolution.needs_clarification,
        clarification=resolution.clarification() if resolution.needs_clarification else "",
        extraction=extraction,
    )


def _interpretation(extraction: entities.Extraction, resolution: resolver.Resolution) -> str:
    measure = {
        "sum": "total", "avg": "average", "count": "number of rows for",
    }.get(extraction.aggregate, extraction.aggregate)
    metric = "sales (EUR)" if extraction.metric == "sales_eur" else extraction.metric

    sentence = f"I will compute the {measure} {metric}"
    if extraction.group_by:
        sentence += " broken down by " + ", ".join(extraction.group_by)

    filters = [
        f"{column} = {', '.join(sorted(set(values)))}"
        for column, values in resolution.resolved.items()
    ]
    if extraction.years:
        filters.append("year = " + ", ".join(str(y) for y in extraction.years))
    if filters:
        sentence += ", restricted to " + "; ".join(filters)

    if extraction.top_n:
        sentence += f", keeping the {'bottom' if extraction.ascending else 'top'} {extraction.top_n}"

    return sentence + "."


# ---------------------------------------------------------------------------
# Phase 2 — run what was approved
# ---------------------------------------------------------------------------

def execute(question: str, *, skip_clarification: bool = False) -> Answer:
    started = time.monotonic()

    extraction = entities.extract(question)
    resolution = resolver.resolve(extraction.entities)

    if resolution.needs_clarification and not skip_clarification:
        return Answer(
            question=question,
            ok=False,
            error=resolution.clarification(),
            elapsed_s=time.monotonic() - started,
        )

    sql = planner.generate(extraction, resolution)
    guard = sql_guard.validate(sql)
    if not guard.ok:
        logger.warning("query refused by guard: %s", guard.error)
        return Answer(
            question=question,
            ok=False,
            sql=sql,
            error=f"query refused: {guard.error}",
            elapsed_s=time.monotonic() - started,
        )

    try:
        frame = schema.query(guard.sql)
    except Exception as exc:
        return Answer(
            question=question,
            ok=False,
            sql=guard.sql,
            error=f"execution failed: {exc}",
            elapsed_s=time.monotonic() - started,
        )

    answer = Answer(
        question=question,
        ok=True,
        sql=guard.sql,
        columns=[str(c) for c in frame.columns],
        rows=frame.values.tolist(),
        row_count=len(frame),
        chart=suggest_chart(frame, extraction),
        elapsed_s=time.monotonic() - started,
    )
    answer.narration = narrate(question, frame)
    return answer


# ---------------------------------------------------------------------------
# Narration and chart choice
# ---------------------------------------------------------------------------

NARRATION_PROMPT = """You state what a result table shows, in two sentences at most.
Quote the figures exactly as given — never round, never extrapolate, never add a
figure that is not in the table. If the table is empty, say so plainly."""


def narrate(question: str, frame: pd.DataFrame) -> str:
    if llm.available():
        try:
            preview = frame.head(20).to_markdown(index=False)
            return llm.complete(
                NARRATION_PROMPT, f"Question: {question}\n\nResult:\n{preview}"
            ).text.strip()
        except Exception:
            logger.warning("narration failed, using template", exc_info=True)

    return _narrate_template(frame)


def _narrate_template(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No row matches this question."

    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if len(frame) == 1 and len(numeric) == 1:
        return f"{numeric[0].replace('_', ' ')}: {frame[numeric[0]].iloc[0]:,.0f}."

    if numeric and len(frame.columns) > 1:
        label_col = frame.columns[0]
        metric_col = numeric[-1]
        top = frame.iloc[0]
        return (
            f"{len(frame)} rows. Leading {label_col}: {top[label_col]} "
            f"({metric_col.replace('_', ' ')} {top[metric_col]:,.0f})."
        )

    return f"{len(frame)} rows returned."


def suggest_chart(frame: pd.DataFrame, extraction: entities.Extraction) -> dict[str, str] | None:
    """Pick a chart type from the shape of the result, not from the question.

    A question can ask for a "trend" and come back with one row; the table is
    what the reader will look at, so the table decides.
    """
    if frame.empty or len(frame.columns) < 2 or len(frame) < 2:
        return None

    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        return None

    label = next((c for c in frame.columns if c not in numeric), frame.columns[0])
    value = numeric[-1]

    if label in ("period", "quarter", "year"):
        kind = "line"
    elif len(frame) <= 8 and extraction.query_type != "ranking":
        kind = "pie"
    else:
        kind = "bar"

    return {"type": kind, "x": str(label), "y": str(value)}
