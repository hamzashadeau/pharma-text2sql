"""HTTP surface — two endpoints, because the gate is two phases.

    uvicorn pharma_text2sql.api:app --reload

``POST /explain`` understands the question and returns what it believes it was
asked. ``POST /execute`` runs it. A client that calls ``/execute`` directly
still gets the clarification behaviour, but it skips the human confirmation —
which is a choice the caller makes explicitly rather than one the API makes
for them.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__, llm, schema
from . import pipeline as _pipeline

app = FastAPI(
    title="pharma-text2sql",
    description="Natural-language questions over a pharmaceutical sales table, with guardrails.",
    version=__version__,
)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ExplainResponse(BaseModel):
    question: str
    interpretation: str
    resolved: dict[str, list[str]]
    needs_clarification: bool
    clarification: str


class ExecuteResponse(BaseModel):
    question: str
    ok: bool
    sql: str
    narration: str
    columns: list[str]
    rows: list[list]
    row_count: int
    chart: dict[str, str] | None
    error: str
    elapsed_s: float


@app.get("/health")
def health() -> dict:
    """Enough to tell a load balancer and a human what state the service is in."""
    try:
        table = schema.get()
        return {
            "status": "ok",
            "rows": table.row_count,
            "columns": len(table.columns),
            "llm": "configured" if llm.available() else "offline (deterministic planner)",
        }
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


@app.get("/schema")
def describe() -> dict:
    table = schema.get()
    return {
        "row_count": table.row_count,
        "columns": [{"name": c.name, "type": c.type} for c in table.columns],
    }


@app.post("/explain", response_model=ExplainResponse)
def explain(body: Question) -> ExplainResponse:
    """Phase one. No SQL is executed here — that is the point."""
    result = _pipeline.explain(body.question)
    return ExplainResponse(
        question=result.question,
        interpretation=result.interpretation,
        resolved=result.resolved,
        needs_clarification=result.needs_clarification,
        clarification=result.clarification,
    )


@app.post("/execute", response_model=ExecuteResponse)
def execute(body: Question) -> ExecuteResponse:
    """Phase two. Runs what phase one described."""
    answer = _pipeline.execute(body.question)
    return ExecuteResponse(
        question=answer.question,
        ok=answer.ok,
        sql=answer.sql,
        narration=answer.narration,
        columns=answer.columns,
        rows=[[_jsonable(cell) for cell in row] for row in answer.rows],
        row_count=answer.row_count,
        chart=answer.chart,
        error=answer.error,
        elapsed_s=round(answer.elapsed_s, 3),
    )


def _jsonable(cell):
    """numpy scalars do not serialise; everything else passes through."""
    if hasattr(cell, "item"):
        try:
            return cell.item()
        except Exception:
            return str(cell)
    return cell
