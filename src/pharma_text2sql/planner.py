"""Step 3 — write the SQL.

Two generators behind one signature:

* **the model**, given the schema and the *already resolved* entity values, so
  it never has to guess how the database spells "the UK";
* **a template planner**, used when no key is configured, which builds the
  query from the structured intent.

Whichever writes it, the query then goes through :mod:`sql_guard` before it is
allowed anywhere near the database. The guard is not a formality for the
template path: it is what makes the model path safe to ship, and running both
through it means the safety net is exercised on every single run.
"""

from __future__ import annotations

import logging

from . import llm, schema
from .config import settings
from .entities import Extraction
from .resolver import Resolution

logger = logging.getLogger(__name__)

NUMERIC = {"units", "sales_eur", "pack_size", "year"}

SYSTEM_PROMPT = """You write DuckDB SQL for a pharmaceutical sales table.

Rules:
- Return ONLY the SQL. No explanation, no code fence.
- One SELECT statement. Never write, never attach, never read files.
- Use ONLY the columns listed below.
- Use the resolved entity values given to you verbatim in WHERE clauses.
- Always add an explicit ORDER BY when the question implies a ranking.
- Keep the result small: aggregate rather than returning raw rows."""


def generate(extraction: Extraction, resolution: Resolution) -> str:
    """Return a SQL string for the question. Not yet validated."""
    if llm.available():
        try:
            return _generate_llm(extraction, resolution)
        except Exception:
            logger.warning("model generation failed, using template planner", exc_info=True)
    return _generate_template(extraction, resolution)


def _generate_llm(extraction: Extraction, resolution: Resolution) -> str:
    resolved = "\n".join(
        f"  {column}: {', '.join(repr(v) for v in values)}"
        for column, values in resolution.resolved.items()
    ) or "  (none)"

    user = (
        f"{schema.get().describe()}\n\n"
        f"Resolved entity values:\n{resolved}\n\n"
        f"Question: {extraction.question}"
    )
    sql = llm.complete(SYSTEM_PROMPT, user).text.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0]
    return sql.strip()


# ---------------------------------------------------------------------------
# Template planner
# ---------------------------------------------------------------------------

def _generate_template(extraction: Extraction, resolution: Resolution) -> str:
    table = settings.table_name
    metric = extraction.metric if extraction.metric in NUMERIC else "sales_eur"

    if extraction.aggregate == "count":
        measure = "count(*)"
        alias = "row_count"
    elif extraction.aggregate == "avg":
        measure = f'avg("{metric}")'
        alias = f"avg_{metric}"
    else:
        measure = f'sum("{metric}")'
        alias = f"total_{metric}"

    group_by = [c for c in extraction.group_by if c in set(schema.get().column_names)]

    select_parts = [f'"{c}"' for c in group_by] + [f"{measure} AS {alias}"]
    sql = [f"SELECT {', '.join(select_parts)}", f"FROM {table}"]

    where = _where(extraction, resolution)
    if where:
        sql.append("WHERE " + "\n  AND ".join(where))

    if group_by:
        sql.append("GROUP BY " + ", ".join(f'"{c}"' for c in group_by))

        if extraction.query_type == "trend" or any(
            c in group_by for c in ("year", "quarter", "period")
        ):
            order_col = next(c for c in ("period", "quarter", "year") if c in group_by)
            sql.append(f'ORDER BY "{order_col}"')
        else:
            sql.append(f"ORDER BY {alias} {'ASC' if extraction.ascending else 'DESC'}")

    if extraction.top_n:
        sql.append(f"LIMIT {extraction.top_n}")

    return "\n".join(sql)


def _where(extraction: Extraction, resolution: Resolution) -> list[str]:
    clauses: list[str] = []

    for column, values in resolution.resolved.items():
        unique = sorted(set(values))
        if len(unique) == 1:
            clauses.append(f'"{column}" = {_literal(unique[0])}')
        else:
            joined = ", ".join(_literal(v) for v in unique)
            clauses.append(f'"{column}" IN ({joined})')

    if extraction.years:
        if len(extraction.years) == 1:
            clauses.append(f'"year" = {extraction.years[0]}')
        else:
            joined = ", ".join(str(y) for y in extraction.years)
            clauses.append(f'"year" IN ({joined})')

    return clauses


def _literal(value: str) -> str:
    """Single-quoted SQL literal, doubling embedded quotes.

    Values come from the database itself, not from the user — but a product
    label containing an apostrophe would still break the query, and quoting it
    properly costs one line.
    """
    return "'" + str(value).replace("'", "''") + "'"
