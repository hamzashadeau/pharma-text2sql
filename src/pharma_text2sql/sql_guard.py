"""Guardrail: a generated query is read-only, single-statement and capped.

The SQL arriving here was written by a language model steered by whatever the
user typed. Three things are checked, in order of how badly they fail:

1. **Parse.** The statement is parsed with sqlglot. Anything that is not a
   single ``SELECT`` is refused — that covers ``DROP``, ``INSERT``, stacked
   statements and the attach/copy family DuckDB also understands.
2. **Tables.** Only the one table is reachable, so a query cannot read
   ``read_csv('/etc/passwd')`` or attach another database.
3. **Shape.** A ``LIMIT`` is forced, so a question that would return millions
   of rows cannot take the service down.

Keyword scanning is done on the **AST**, not on the raw text, because the text
contains data: a molecule named ``AMINODIHYDROPHTHALASINDIONE`` contains
"DROP", and an eye-drop product contains it twice. A regex denylist refuses
those perfectly legitimate questions — which is exactly the kind of bug that
makes people disable the guardrail altogether.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .config import settings

DIALECT = "duckdb"

# Expression types that write, change or reach outside the query.
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Command,   # DuckDB ATTACH / COPY / EXPORT / PRAGMA land here
    exp.Set,
)

# Table-valued functions that read from the filesystem or the network.
FORBIDDEN_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "glob", "parquet_scan", "csv_scan", "httpfs",
    "install", "load",
}


@dataclass
class GuardResult:
    ok: bool
    sql: str = ""
    error: str = ""


def validate(sql: str, *, max_rows: int | None = None) -> GuardResult:
    """Validate and normalise a generated query."""
    max_rows = max_rows or settings.max_rows
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return GuardResult(ok=False, error="empty query")

    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except Exception as exc:
        return GuardResult(ok=False, error=f"unparseable SQL: {exc}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return GuardResult(
            ok=False,
            error=f"expected exactly one statement, got {len(statements)}",
        )

    tree = statements[0]

    # A top-level UNION of SELECTs is still a read, and some legitimate
    # questions ("this year next to last year") are naturally written that way.
    if not isinstance(tree, (exp.Select, exp.Union, exp.With)):
        return GuardResult(ok=False, error=f"only SELECT is allowed, got {type(tree).__name__}")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return GuardResult(ok=False, error=f"forbidden statement: {type(node).__name__}")

        if isinstance(node, exp.Func):
            # Anonymous covers functions sqlglot does not model; sql_name covers
            # the ones it does — read_csv parses to a ReadCSV node, not to a
            # function call, and checking only Anonymous would wave it through.
            names = {(node.name or "").lower()}
            try:
                names.add(node.sql_name().lower())
            except Exception:
                pass
            forbidden = names & FORBIDDEN_FUNCTIONS
            if forbidden:
                return GuardResult(ok=False, error=f"forbidden function: {forbidden.pop()}")

        if isinstance(node, exp.Table):
            # An empty name means the source is not a table but an expression —
            # a table function, a file read. Only the one real table is allowed.
            name = (node.name or "").lower()
            if name != settings.table_name:
                return GuardResult(
                    ok=False,
                    error=f"unknown table: {node.name or node.sql(dialect=DIALECT)}",
                )

    # Column allowlist — catches hallucinated columns before DuckDB does, and
    # gives the generator a message it can act on.
    known = {c.lower() for c in _known_columns()}
    if known:
        aliases = {
            (a.alias or "").lower()
            for a in tree.find_all(exp.Alias)
        } | {
            (t.alias or "").lower() for t in tree.find_all(exp.Table)
        }
        for col in tree.find_all(exp.Column):
            name = (col.name or "").lower()
            if name and name not in known and name not in aliases:
                return GuardResult(ok=False, error=f"unknown column: {col.name}")

    # Force a ceiling on the result size.
    limit = tree.args.get("limit")
    if limit is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        try:
            if int(limit.expression.name) > max_rows:
                tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (AttributeError, ValueError):
            tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return GuardResult(ok=True, sql=tree.sql(dialect=DIALECT, pretty=True))


def _known_columns() -> list[str]:
    # Imported lazily: the guard is unit-tested without a dataset on disk.
    try:
        from . import schema

        return schema.get().column_names
    except Exception:
        return []
